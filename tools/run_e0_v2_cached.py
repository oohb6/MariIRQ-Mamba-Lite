#!/usr/bin/env python3
"""E0/V2 pipeline: NPZ -> Adapt WTCM cache / Lite online WICM -> train -> eval -> summary.

Use --phase all for the complete sequence; existing checkpoints/results are never
silently overwritten. E0 and V2 are independent experiments with fresh weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.verify_precomputed_cache import check_split

METRICS = ('Accuracy', 'Precision', 'Recall', 'F1-score', 'Top-5')


def run(cmd, log=None):
    print('>>>', ' '.join(map(str, cmd)), flush=True)
    if log is None:
        subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as out:
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            print(line, end='', flush=True)
            out.write(line)
        status = p.wait()
        if status:
            raise subprocess.CalledProcessError(status, cmd)


def check_inputs(data):
    labels = {}
    for split in ('train', 'valid', 'test'):
        source = data / f'{split}.npz'
        if not source.is_file():
            raise FileNotFoundError(f'Missing {source}. Prepare raw traces into frozen NPZ splits first; see README.md')
        with np.load(source, allow_pickle=False) as z:
            X = z['X']
            y = z['y']
        if X.ndim != 3 or X.shape[1:] != (5000, 2) or X.dtype.kind not in 'f':
            raise ValueError(f'{source}: X must be float (N,5000,2), got {X.shape} {X.dtype}')
        if y.ndim != 1 or y.dtype.kind not in 'iu' or len(y) != len(X) or len(y) == 0:
            raise ValueError(f'{source}: y must be nonempty 1-D integer matching X')
        if not np.isfinite(X).all():
            raise ValueError(f'{source}: non-finite trace values')
        labels[split] = np.unique(y)
        print(f'[DATA] {source}: X={X.shape}; classes present={len(labels[split])}')
    train_classes = labels['train']
    if not np.array_equal(train_classes, np.arange(len(train_classes))):
        raise ValueError(f'{data}: TRAIN IDs must be contiguous 0..K-1; preserve original mapping')
    for split in ('valid', 'test'):
        if not np.isin(labels[split], train_classes).all():
            raise ValueError(f'{data}: {split} has class IDs missing in TRAIN')
    return len(train_classes)


def meta_hash(data):
    h = hashlib.sha256()
    for split in ('train', 'valid'):
        h.update((data / 'wtcm_cache' / f'{split}.meta.json').read_bytes())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lite_online_signature(data):
    """Track the raw NPZs and implementation for Lite without WICM cache."""
    h = hashlib.sha256()
    for path in (data / 'train.npz', data / 'valid.npz',
                 ROOT / 'mariirq/wicm.py', ROOT / 'train_lite.py'):
        h.update(path.name.encode('utf-8'))
        h.update(file_sha256(path).encode('ascii'))
    return h.hexdigest()


def prepare(variant, data, a):
    if variant == 'v2' and not (data / 'train.npz').exists():
        # This transformation is fitted ONLY on original E0 train. No target/test fits.
        if data.exists():
            raise FileExistsError(f'Partial V2 folder: {data}; inspect it before resuming')
        run([sys.executable, '-u', 'tools/prepare_robust_v2.py', 'fit',
             '--raw', a.v2_raw, '--baseline-ref', a.e0_data,
             '--out', data, '--alpha', a.alpha])
    check_inputs(data)
    if variant == 'v2':
        ref = data / 'robust_v2.json'
        if not ref.is_file():
            raise FileNotFoundError(f'V2 needs its frozen robust_v2.json: {ref}')
        obj = json.loads(ref.read_text(encoding='utf-8'))
        if obj.get('version') != 'robust_v2_trace_relative_1':
            raise ValueError(f'Unexpected V2 method version in {ref}')
        if abs(float(obj.get('alpha', -1)) - a.alpha) > 1e-12:
            raise ValueError(f'{ref}: alpha={obj.get("alpha")} differs from --alpha={a.alpha}. Use new experiment and correct method parameters.')
        for filename, digest in obj.get('reference_files_sha256', {}).items():
            actual = a.e0_data / filename
            if not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest() != digest:
                raise ValueError(f'V2 frozen E0 reference changed/missing: {actual}')
        print(f'[V2] alpha={obj["alpha"]}; original E0 frozen config checked')


def build_or_verify_cache(data, model, workers):
    if model == 'lite':
        print('[ONLINE WICM] Lite computes features during training/testing; no cache required.', flush=True)
        return
    if model != 'adapt':
        raise ValueError(f'Unknown model: {model}')
    cache = data / 'wtcm_cache'
    for split in ('train', 'valid'):
        names = [cache / f'{split}.meta.json', cache / f'{split}_matrix.npy', cache / f'{split}_index.npy']
        partials = [cache / f'{split}_matrix.partial.npy', cache / f'{split}_index.partial.npy', cache / f'{split}.meta.json.tmp']
        if all(f.is_file() for f in names) and not any(f.exists() for f in partials):
            check_split(data, 'adapt', split)
            continue
        if any(f.exists() for f in names + partials):
            raise FileExistsError(f'{cache}/{split} is partial/incomplete. Inspect files; do not overwrite and mix features.')
        with np.load(data / f'{split}.npz', allow_pickle=False) as z:
            n_samples = len(z['y'])
        needed = n_samples * 35 * 4998 * 4 + 256 * 1024 * 1024
        available = shutil.disk_usage(data).free
        if needed > available:
            raise OSError(f'{data}: {split} Adapt cache needs approximately {needed/1024**3:.2f} GiB, '
                          f'but only {available/1024**3:.2f} GiB free')
        run([sys.executable, '-u', 'tools/precompute_wtcm.py',
             '--data_path', data, '--output_dir', cache,
             '--splits', split, '--workers', workers])
        check_split(data, 'adapt', split)


def train_one(variant, data, model, seed, a):
    out = a.checkpoint_root / a.experiment / variant / model / f'seed_{seed}'
    log = a.log_root / a.experiment / variant / model / f'train_{seed}.log'
    lite_online = model == 'lite'
    if lite_online:
        # Never silently reuse an old cached Lite run as an online Lite run.
        cfg = dict(data_path=str(data.resolve()), model=model, variant=variant, seed=seed,
                   epochs=a.epochs, patience=a.patience, batch_size=a.batch_size,
                   feature_mode='online_wicm', source_signature_sha256=lite_online_signature(data))
    else:
        # Preserve the old marker schema when explicitly opting into cached Lite.
        cfg = dict(data_path=str(data.resolve()), model=model, variant=variant, seed=seed,
                   epochs=a.epochs, patience=a.patience, batch_size=a.batch_size,
                   cache_meta_sha256=meta_hash(data))
    marker = out / 'train_done.json'
    summary, ckpt = out / 'training_summary.json', out / 'max_f1.pth'
    if marker.is_file() and summary.is_file() and ckpt.is_file() and summary.stat().st_size > 0 and ckpt.stat().st_size > 0:
        if json.loads(marker.read_text(encoding='utf-8')) == cfg:
            print(f'[SKIP TRAIN VERIFIED] {variant}/{model}/seed_{seed}')
            return ckpt
        raise ValueError(f'Existing training marker has DIFFERENT configuration: {out}')
    if marker.exists() or ckpt.exists() or summary.exists():
        raise FileExistsError(f'Partial/older training output at {out}. Never resume from an arbitrary best checkpoint. Use a new --experiment name.')
    out.mkdir(parents=True, exist_ok=True)
    if model == 'adapt':
        script = 'train_adapt_cached.py'
        cache = data / 'wtcm_cache'
    else:
        script = 'train_lite.py'
        cache = None
    cmd = [sys.executable, '-u', script, '--data_path', data]
    if cache is not None:
        cmd += ['--cache_dir', cache]
    cmd += ['--output_dir', out,
            '--seed', seed, '--epochs', a.epochs, '--patience', a.patience,
            '--batch_size', a.batch_size,
            '--num_workers', a.adapt_workers if model == 'adapt' else a.lite_workers]
    if model == 'adapt':
        cmd += ['--prefetch_factor', a.prefetch_factor]
    print(f'[TRAIN MODE] {variant}/{model}: {"WICM ONLINE" if lite_online else "CACHED FEATURES"}', flush=True)
    run(cmd, log)
    if not ckpt.is_file() or not summary.is_file():
        raise RuntimeError(f'Trainer returned without checkpoint and summary: {out}')
    marker.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'[TRAIN COMPLETE] {out}', flush=True)
    return ckpt


def test_file(variant, data, condition, a):
    if condition == 'cw_test':
        return data / 'test.npz'
    base = a.e0_test_root if variant == 'e0' else a.v2_test_root
    return base / condition / 'test.npz'


def evaluate_one(variant, data, model, seed, condition, a):
    ckpt = a.checkpoint_root / a.experiment / variant / model / f'seed_{seed}' / 'max_f1.pth'
    if not ckpt.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {ckpt}')
    src = test_file(variant, data, condition, a)
    if not src.is_file():
        raise FileNotFoundError(f'Test condition {condition} missing: {src} (prepare with frozen E0 mapping/V2 transform)')
    out = a.result_root / a.experiment / variant / condition / model / f'seed_{seed}.json'
    mark = out.with_name(out.stem + '.eval_done.json')
    def file_hash(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    cfg = dict(model=model, variant=variant, seed=seed, condition=condition,
               test_file=str(src.resolve()), test_sha256=file_hash(src),
               checkpoint=str(ckpt.resolve()), checkpoint_sha256=file_hash(ckpt),
               batch_size=a.batch_size)
    if out.exists() or mark.exists():
        if out.is_file() and out.stat().st_size and mark.is_file():
            if json.loads(mark.read_text(encoding='utf-8')) == cfg:
                json.loads(out.read_text(encoding='utf-8'))
                print(f'[SKIP EVAL VERIFIED] {out}')
                return
        raise FileExistsError(f'Existing partial/stale evaluation in {out.parent}; use a new --experiment name')
    cmd = [sys.executable, '-u', f'test_{model}.py',
           '--test_file', src, '--checkpoint', ckpt, '--result_file', out,
           '--batch_size', a.batch_size,
           '--num_workers', a.adapt_workers if model == 'adapt' else a.lite_workers]
    run(cmd, a.log_root / a.experiment / variant / model / f'test_{condition}_{seed}.log')
    if not out.is_file():
        raise RuntimeError(f'Test failed to create {out}')
    mark.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_summary(variants, a):
    rows = []
    for variant in variants:
        for model in a.models:
            for condition in a.conditions:
                reports = []
                for seed in a.seeds:
                    p = a.result_root / a.experiment / variant / condition / model / f'seed_{seed}.json'
                    if not p.is_file(): raise FileNotFoundError(f'Cannot summarize missing {p}')
                    reports.append(json.loads(p.read_text(encoding='utf-8')))
                row = dict(variant=variant, model=model, condition=condition, n_seeds=len(reports))
                for metric in METRICS:
                    vals = [float(r[metric]) for r in reports]
                    row[metric + '_mean'] = round(statistics.mean(vals), 4)
                    row[metric + '_std'] = round(statistics.stdev(vals), 4) if len(vals)>1 else 0.
                rows.append(row)
                print(f'[SUMMARY] {variant}/{model}/{condition}: Acc={row["Accuracy_mean"]} +/- {row["Accuracy_std"]}; F1={row["F1-score_mean"]} +/- {row["F1-score_std"]}', flush=True)
    target = a.result_root / a.experiment / 'summary_mean_std.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'[SAVED] {target}', flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('e0', 'v2', 'both'), default='e0')
    p.add_argument('--phase', choices=('all', 'prepare', 'check', 'cache', 'train', 'eval', 'summary'), default='all')
    p.add_argument('--e0-data', type=Path, default=Path('data/processed/chrome_cw'))
    p.add_argument('--v2-data', type=Path, default=Path('data/processed/robust_v2/chrome_cw'))
    p.add_argument('--v2-raw', type=Path, default=Path('datasets/raw/closed_world'))
    p.add_argument('--alpha', type=float, default=1.0, help='V2 fit ONLY if V2 train data missing; new experiment required for different alpha')
    p.add_argument('--e0-test-root', type=Path, default=Path('data/processed/e0_tests'))
    p.add_argument('--v2-test-root', type=Path, default=Path('data/processed/v2_tests'))
    p.add_argument('--models', nargs='+', choices=('adapt', 'lite'), default=['adapt', 'lite'])
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(2024, 2029)))
    p.add_argument('--conditions', nargs='+', default=['cw_test'])
    p.add_argument('--cache-workers', type=int, default=8)
    p.add_argument('--adapt-workers', type=int, default=4)
    p.add_argument('--lite-workers', type=int, default=4)
    p.add_argument('--lite-feature-mode', choices=('online',), default='online', help='Lite only supports the original online WICM workflow')
    p.add_argument('--prefetch-factor', type=int, default=1)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--experiment', default='production', help='Isolate smoke/production and any changed training settings')
    p.add_argument('--checkpoint-root', type=Path, default=Path('checkpoints/e0_v2_cached'))
    p.add_argument('--result-root', type=Path, default=Path('results/e0_v2_cached'))
    p.add_argument('--log-root', type=Path, default=Path('logs/e0_v2_cached'))
    a = p.parse_args()
    if not (0 <= a.alpha <= 1): p.error('--alpha must be in [0,1]')
    if a.epochs < 1 or a.patience < 1 or a.cache_workers < 1 or a.batch_size < 1: p.error('epochs, patience, cache-workers, batch-size must be positive')
    if a.adapt_workers < 0 or a.lite_workers < 0: p.error('worker counts must be >=0')
    if a.experiment in ('.', '..') or '/' in a.experiment or '\\' in a.experiment: p.error('--experiment must be one simple directory name')
    a.seeds = list(dict.fromkeys(a.seeds))
    a.conditions = list(dict.fromkeys(a.conditions))
    a.models = list(dict.fromkeys(a.models))
    for name in ('e0_data', 'v2_data', 'v2_raw', 'e0_test_root', 'v2_test_root',
                 'checkpoint_root', 'result_root', 'log_root'):
        setattr(a, name, (ROOT / getattr(a, name)).resolve())
    return a


def main():
    a = parse()
    os.chdir(ROOT)
    variants = ['e0', 'v2'] if a.variant == 'both' else [a.variant]
    dataset = {'e0': a.e0_data, 'v2': a.v2_data}
    phases = ('prepare', 'cache', 'train', 'eval', 'summary') if a.phase == 'all' else (a.phase,)
    print(f'PIPELINE variants={variants} phases={phases} models={a.models} seeds={a.seeds} conditions={a.conditions}', flush=True)
    for phase in phases:
        if phase == 'summary':
            write_summary(variants, a)
            continue
        for variant in variants:
            data = dataset[variant]
            if phase in ('prepare', 'check'):
                if phase == 'prepare':
                    prepare(variant, data, a)
                else:
                    check_inputs(data)
                if phase == 'check':
                    for model in a.models:
                        for split in ('train','valid'):
                            if model == 'adapt':
                                check_split(data, model, split)
                            else:
                                print('[ONLINE WICM] No Lite cache is required.', flush=True)
                elif a.phase == 'all':
                    # Preflight every target before expensive multi-seed training.
                    for condition in a.conditions:
                        test_src = test_file(variant, data, condition, a)
                        if not test_src.is_file():
                            raise FileNotFoundError(f'Prepare requested {condition} before training: {test_src}')
            elif phase == 'cache':
                check_inputs(data)
                for model in a.models: build_or_verify_cache(data, model, a.cache_workers)
            elif phase == 'train':
                check_inputs(data)
                for model in a.models:
                    if model == 'adapt':
                        for split in ('train','valid'): check_split(data, model, split)
                    for seed in a.seeds: train_one(variant, data, model, seed, a)
            elif phase == 'eval':
                for model in a.models:
                    for seed in a.seeds:
                        for condition in a.conditions: evaluate_one(variant, data, model, seed, condition, a)
    print('PIPELINE DONE', flush=True)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Isolated per-trace robust alignment for MariIRQ V2.

Uses ONLY CW training statistics to fix anchors. Each individual trace gets the
same predefined feature transform; the target DOMAIN population is never used.
Raw samples are transformed before clipping (critical for Chrome 92).
Preserves frozen class mapping and exact CW train/valid/test split.

Run from project root.  See README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'tools'))
from prepare_baseline import L, canon, load_raw, map_y, to_x

TINY = np.float32(1e-6)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def raw_to_unbounded(trace, train_scaler):
    """Legacy scale without destructive clipping. Do NOT start from old NPZ."""
    raw = np.asarray(trace, dtype=np.float32).reshape(-1)[:L]
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite original IRQ sample')
    out = raw.copy()
    live = out != 0
    lo = float(train_scaler['raw_nonzero_min'])
    hi = float(train_scaler['raw_nonzero_max'])
    if not hi > lo:
        raise ValueError('Invalid frozen CW TRAIN scaler')
    out[live] = (out[live] - lo) / (hi - lo) * 1500.0
    return out


def trace_stats(unbounded, valid_mask=None):
    valid = unbounded[valid_mask] if valid_mask is not None else unbounded[unbounded != 0]
    if not len(valid):
        return None
    q25, med, q75 = np.quantile(valid.astype(np.float64), [.25, .5, .75])
    return float(med), float(max(q75 - q25, 0.0))


def align_one(trace, frozen, robust):
    """Align nonzero values individually; never use statistics across TEST traces."""
    raw = np.asarray(trace, dtype=np.float32).reshape(-1)[:L]
    v = raw_to_unbounded(raw, frozen)
    live = raw != 0
    if live.any():
        med, iqr = trace_stats(v, live)
        ref_m = float(robust['reference_median'])
        ref_q = float(robust['reference_iqr'])
        floor = float(robust['iqr_floor'])
        alpha = float(robust['alpha'])
        # Cap extreme gain to avoid amplifying almost-constant traces.
        gain = np.clip(ref_q / max(iqr, floor),
                       float(robust['gain_min']), float(robust['gain_max']))
        centered = ref_m + (v[live].astype(np.float64) - med) * gain
        combined = (1.0 - alpha) * v[live].astype(np.float64) + alpha * centered
        v[live] = np.clip(combined, float(TINY), 1500.0).astype(np.float32)
    return v


def make_x(traces, frozen, robust):
    n = len(traces)
    X = np.zeros((n, L, 2), dtype=np.float32)
    timeline = np.arange(L, dtype=np.float32) * np.float32(.001)
    timeline[0] = TINY
    clipped = 0
    for idx, tr in enumerate(traces):
        length = min(len(np.asarray(tr).reshape(-1)), L)
        if length:
            X[idx, :length, 0] = timeline[:length]
            X[idx, :length, 1] = align_one(tr, frozen, robust)
        if (idx + 1) % 1000 == 0:
            print(f'Converted {idx+1}/{n}', flush=True)
        clipped += int(np.count_nonzero(X[idx, :length, 1] >= 1499.999))
    print(f'Built {n} V2 traces (N,5000,2); values at upper bound={clipped}', flush=True)
    return X


def verify_old_samples(traces, original_y, baseline_ref, split, src_idx):
    """Fail fast if input raw PKL ordering differs from the stored baseline NPZ."""
    old = baseline_ref / (split + '.npz')
    with np.load(old, allow_pickle=False) as z:
        prior_x, prior_y = z['X'], z['y']
    if len(prior_y) != len(src_idx):
        raise ValueError(f'Raw/split mismatch: {old}: {len(prior_y)} != {len(src_idx)}')
    if not np.array_equal(prior_y, original_y[src_idx]):
        raise ValueError(f'Label ORDER mismatch: {old}; stop (do not re-number)')
    for p in sorted(set((0, len(src_idx)//2, len(src_idx)-1))):
        r = int(src_idx[p])
        old_rebuilt = to_x([traces[r]], json.loads((baseline_ref/'scaler.json').read_text()))[0]
        # Exact baseline scaler and array are expected; tolerate float32 rounding only.
        if not np.allclose(prior_x[p], old_rebuilt, atol=3e-4, rtol=1e-6):
            err = float(np.max(np.abs(prior_x[p] - old_rebuilt)))
            raise ValueError(f'RAW SOURCE MISMATCH for {old} index={p}: max error {err:.5g}')
    return True


def save(path, X, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing overwrite: {path}')
    np.savez_compressed(path, X=X, y=y)
    print(f'[SAVED] {path}: X={X.shape} y={y.shape} unique_y={len(np.unique(y))}',flush=True)


def _check_reference(ref):
    required = ['mapping.json', 'scaler.json', 'split_indices.npz',
                'train.npz', 'valid.npz', 'test.npz']
    for x in required:
        if not (ref/x).is_file():
            raise FileNotFoundError(ref/x)


def fit(args):
    base = args.baseline_ref.resolve()
    _check_reference(base)
    if args.out.exists():
        raise FileExistsError('Use a NEW V2 output directory; refusing to mix experiments: '+str(args.out))
    mapping = json.loads((base/'mapping.json').read_text(encoding='utf-8'))
    frozen = json.loads((base/'scaler.json').read_text(encoding='utf-8'))
    raw, labels = load_raw(args.raw)
    y = map_y(labels, mapping)
    with np.load(base/'split_indices.npz', allow_pickle=False) as z:
        ids = {k: np.asarray(z[k],dtype=np.int64) for k in ('train_idx','valid_idx','test_idx')}
    train, valid, test = [ids[k] for k in ('train_idx','valid_idx','test_idx')]
    combined = np.concatenate([train,valid,test])
    if len(combined)!=len(y) or sorted(combined.tolist())!=list(range(len(y))):
        raise ValueError('Legacy split indices do not match the provided raw source')
    for name, ix in [('train',train),('valid',valid),('test',test)]:
        verify_old_samples(raw, y, base, name, ix)
    print('[PASS] Exact class order, split indices and sampled old/raw values match CW baseline',flush=True)
    train_m, train_q = [], []
    for idx in train:
        tr = np.asarray(raw[int(idx)], dtype=np.float32).reshape(-1)[:L]
        stat = trace_stats(raw_to_unbounded(tr, frozen), tr != 0)
        if stat is not None:
            train_m.append(stat[0]); train_q.append(stat[1])
    if len(train_m)<len(train)*.9:
        raise ValueError('Unusually many empty IRQ traces in CW train')
    if np.median(train_q) <= 0:
        raise ValueError('CW train per-trace IQR is zero; robust scale unavailable')
    robust = dict(
        version='robust_v2_trace_relative_1',
        alpha=float(args.alpha),
        reference_median=float(np.median(train_m)),
        reference_iqr=float(np.median(train_q)),
        iqr_floor=float(max(5., np.percentile(train_q, 10))),
        gain_min=.25, gain_max=4., seq_len=L,
        anchor_fitted_on='ONLY original CW TRAIN per-trace stats',
        transform='unbounded frozen raw scaler -> individual median/IQR alignment -> final [epsilon,1500] clip',
        baseline_reference=str(base),
        reference_files_sha256={n:sha256(base/n) for n in ('mapping.json','scaler.json','split_indices.npz')},
        raw_fit_source=str(args.raw.resolve()),
        train_stats_count=len(train_m),
    )
    print('[REFERENCE]', json.dumps({k:robust[k] for k in ('alpha','reference_median','reference_iqr','iqr_floor')},indent=2),flush=True)
    X = make_x(raw, frozen, robust)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, ix in [('train',train),('valid',valid),('test',test)]:
        save(args.out/(name+'.npz'), X[ix], y[ix])
    (args.out/'mapping.json').write_bytes((base/'mapping.json').read_bytes())
    (args.out/'scaler.json').write_bytes((base/'scaler.json').read_bytes())
    (args.out/'split_indices.npz').write_bytes((base/'split_indices.npz').read_bytes())
    (args.out/'robust_v2.json').write_text(json.dumps(robust,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('[FIT FINISHED] V2 train/valid/test independently saved. Baseline untouched.',flush=True)


def test(args):
    ref=args.reference.resolve()
    robust=json.loads((ref/'robust_v2.json').read_text(encoding='utf-8'))
    base=Path(robust['baseline_reference'])
    for fname,digest in robust['reference_files_sha256'].items():
        if sha256(base/fname)!=digest:
            raise ValueError(f'Frozen training source changed since V2 FIT: {fname}')
    for name in ('mapping.json','scaler.json','split_indices.npz'):
        if sha256(ref/name)!=sha256(base/name):
            raise ValueError(f'V2 frozen reference differs from original CW: {name}')
    mapping=json.loads((ref/'mapping.json').read_text(encoding='utf-8'))
    frozen=json.loads((ref/'scaler.json').read_text(encoding='utf-8'))
    raw,labels=load_raw(args.raw)
    y=map_y(labels,mapping)
    if args.baseline_test:
        with np.load(args.baseline_test,allow_pickle=False) as z:
            prev_x,prev_y=z['X'],z['y']
        if len(prev_y)!=len(y) or not np.array_equal(prev_y,y):
            raise ValueError(f'Label/order mismatch against {args.baseline_test}')
        for p in sorted(set((0,len(y)//2,len(y)-1))):
            original=to_x([raw[p]],frozen)[0]
            if not np.allclose(prev_x[p], original, atol=3e-4, rtol=1e-6):
                raise ValueError(f'Raw source differs from prior processed input at index {p}: {args.baseline_test}')
        print('[PASS] Test source order and sampled RAW conversion match previous test NPZ',flush=True)
    if args.out.exists():
        raise FileExistsError('Use a NEW condition output directory: '+str(args.out))
    X=make_x(raw,frozen,robust)
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'test.npz',X,y)
    meta={
        'raw_test_source':str(args.raw.resolve()), 'reference':str(ref),
        'baseline_test':str(args.baseline_test) if args.baseline_test else None,
        'reference_sha256':sha256(ref/'robust_v2.json'), 'count':len(y),
        'all_test_labels_are_known':True,
    }
    (args.out/'reference.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print('[TEST PREPARATION FINISHED] no target-domain parameter fitting',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='mode',required=True)
    f=s.add_parser('fit');f.add_argument('--raw',type=Path,required=True)
    f.add_argument('--baseline-ref',type=Path,required=True)
    f.add_argument('--out',type=Path,required=True)
    f.add_argument('--alpha',type=float,default=1.0)
    t=s.add_parser('test');t.add_argument('--raw',type=Path,required=True)
    t.add_argument('--reference',type=Path,required=True)
    t.add_argument('--out',type=Path,required=True)
    t.add_argument('--baseline-test',type=Path,help='Original transfer NPZ, verify sample identity/order')
    a=p.parse_args()
    if a.mode=='fit':
        if not 0<=a.alpha<=1:p.error('--alpha must lie in [0,1]')
        fit(a)
    else:test(a)

if __name__=='__main__':main()

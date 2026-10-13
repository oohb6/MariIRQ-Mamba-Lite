#!/usr/bin/env python3
"""Strictly validate already-built feature caches against their source NPZs.

Uses the SAME online feature extractors; no model/GPU required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mariirq.irq_wtcm import IRQCountDataset
from mariirq.irq_wtcm_cached import CachedIRQCountDataset

# Fixed published E0/V2 representation parameters, identical in both variants.
ADAPT = dict(seq_len=5000, maximum_load_time=5., max_matrix_len=4998,
             maximum_cell_number=32, irq_bin_size=75., irq_amp_threshold=None,
             irq_cluster_window=20, log_transform=True)


def check_split(data_path: Path, model: str, split: str, samples: int = 3) -> None:
    source = data_path / f'{split}.npz'
    with np.load(source, allow_pickle=False) as obj:
        X, labels = obj['X'], obj['y']
    if X.ndim != 3 or X.shape[1:] != (5000, 2):
        raise ValueError(f'{source}: X must be (N,5000,2), got {X.shape}')
    if labels.ndim != 1 or len(labels) != len(X) or len(X) == 0:
        raise ValueError(f'{source}: invalid X/y lengths or empty split')
    if not np.isfinite(X).all():
        raise ValueError(f'{source}: non-finite input values')
    if model != 'adapt':
        raise ValueError('Only Adapt uses a feature cache; Lite computes WICM online')
    args = SimpleNamespace(**ADAPT)
    cached = CachedIRQCountDataset(data_path / 'wtcm_cache', split, labels, source, args)
    online = IRQCountDataset(X, labels, args)
    chosen = sorted(set(np.linspace(0, len(X) - 1, min(samples, len(X)), dtype=int).tolist()))
    for i in chosen:
        (cm, ci), cy = cached[i]
        (om, oi), oy = online[i]
        if not np.array_equal(cm, om) or int(ci) != int(oi) or cy != oy:
            raise AssertionError(f'{model}/{split} sample={i}: cache != online features')
    print(f'[VERIFIED] {data_path.name} {model}/{split}: {len(X)} samples, indices={chosen}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=('adapt',), default=['adapt'])
    p.add_argument('--splits', nargs='+', choices=('train', 'valid', 'test'), default=['train', 'valid'])
    p.add_argument('--samples', type=int, default=3)
    args = p.parse_args()
    if args.samples < 1: p.error('--samples must be positive')
    for model in args.models:
        for split in args.splits:
            check_split(args.data_path, model, split, args.samples)

if __name__ == '__main__':
    main()

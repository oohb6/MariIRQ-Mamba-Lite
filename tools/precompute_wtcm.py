#!/usr/bin/env python3
"""Precompute unchanged IRQCountDataset.process_data() outputs into NPY memmaps.

Run outside active training to avoid competing for the instance's CPU quota.
"""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mariirq.irq_wtcm import IRQCountDataset
from mariirq.irq_wtcm_cached import feature_config, source_info, file_sha256


_DATASET = None
_MATRIX_FILE = None
_INDEX_FILE = None


def _work(bounds):
    start, end = bounds
    mat = np.load(_MATRIX_FILE, mmap_mode="r+", allow_pickle=False)
    idx = np.load(_INDEX_FILE, mmap_mode="r+", allow_pickle=False)
    for i in range(start, end):
        m, k = _DATASET.process_data(_DATASET.X[i])
        mat[i] = m
        idx[i] = k
    mat.flush()
    idx.flush()
    return start, end


def main():
    p = argparse.ArgumentParser(description="Build deterministic IRQ-WTCM cache")
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=("train", "valid"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=5000)
    p.add_argument("--maximum_load_time", type=float, default=5.0)
    p.add_argument("--max_matrix_len", type=int, default=4998)
    p.add_argument("--maximum_cell_number", type=int, default=32)
    p.add_argument("--irq_bin_size", type=float, default=75.0)
    p.add_argument("--irq_amp_threshold", type=float, default=None)
    p.add_argument("--irq_cluster_window", type=int, default=20)
    p.add_argument("--log_transform", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        src = args.data_path / f"{split}.npz"
        if not src.is_file():
            raise FileNotFoundError(src)
        with np.load(src, allow_pickle=False) as z:
            X, y = z["X"], z["y"]
        n = len(X)
        cfg = SimpleNamespace(**feature_config(args))
        ds = IRQCountDataset(X, y, cfg)
        shape = (n, 1, args.maximum_cell_number + 3, args.max_matrix_len)
        mf = root / f"{split}_matrix.npy"
        idxf = root / f"{split}_index.npy"
        metaf = root / f"{split}.meta.json"
        partial_m = root / f"{split}_matrix.partial.npy"
        partial_i = root / f"{split}_index.partial.npy"
        if any(f.exists() for f in (mf, idxf, metaf, partial_m, partial_i)):
            raise FileExistsError(f"Existing complete/partial cache for {split} in {root}. Verify or move it; refusing overwrite.")
        if n == 0 or X.ndim != 3 or X.shape[1:] != (args.seq_len, 2):
            raise ValueError(f"Expected nonempty X=(N,{args.seq_len},2), got {X.shape}")
        print(f"Preparing {split}: {n} samples -> {shape} float32 (~{np.prod(shape)*4/(1024**3):.2f} GiB)", flush=True)
        mat = np.lib.format.open_memmap(partial_m, mode="w+", dtype=np.float32, shape=shape)
        indices = np.lib.format.open_memmap(partial_i, mode="w+", dtype=np.int64, shape=(n,))
        del mat, indices
        global _DATASET, _MATRIX_FILE, _INDEX_FILE
        _DATASET, _MATRIX_FILE, _INDEX_FILE = ds, partial_m, partial_i
        workers = min(args.workers, max(1, n))
        ranges = [(n*w//workers, n*(w+1)//workers) for w in range(workers)]
        # fork: readonly dataset arrays are shared via copy-on-write; each worker
        # writes a distinct range of the two cache files.
        if workers == 1:
            for b in ranges:
                print(f"Finished {split}: {_work(b)}", flush=True)
        else:
            with mp.get_context("fork").Pool(processes=workers) as pool:
                for done in pool.imap_unordered(_work, ranges):
                    print(f"Finished {split}: {done}", flush=True)
        # Exact equality against the original online implementation on selected traces.
        cached_m = np.load(partial_m, mmap_mode="r", allow_pickle=False)
        cached_i = np.load(partial_i, mmap_mode="r", allow_pickle=False)
        for j in sorted(set([0, n//2, n-1])) if n else []:
            m, k = ds.process_data(ds.X[j])
            if not np.array_equal(cached_m[j], m) or int(cached_i[j]) != int(k):
                raise AssertionError(f"Cache mismatch: split={split}, sample={j}")
        meta = {
            "feature_config": feature_config(args),
            "source_info": source_info(src, y),
            "source_sha256": file_sha256(src),
            "count": n,
            "shape": list(shape),
            "verified_sample_indices": sorted(set([0, n//2, n-1])) if n else [],
        }
        # Commit atomically: a complete .meta.json is written LAST.
        del cached_m, cached_i
        os.replace(partial_m, mf)
        os.replace(partial_i, idxf)
        tmp_meta = root / f"{split}.meta.json.tmp"
        tmp_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_meta, metaf)
        _DATASET, _MATRIX_FILE, _INDEX_FILE = None, None, None
        print(f"VERIFIED and saved {metaf}", flush=True)


if __name__ == "__main__":
    main()

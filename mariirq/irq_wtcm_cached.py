"""Read exact precomputed IRQ-WTCM features without recomputing per epoch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


FEATURE_ARGS = (
    "seq_len", "maximum_load_time", "max_matrix_len",
    "maximum_cell_number", "irq_bin_size", "irq_amp_threshold",
    "irq_cluster_window", "log_transform",
)


def feature_config(args):
    return {key: getattr(args, key) for key in FEATURE_ARGS}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_info(npz_path, labels):
    s = Path(npz_path).stat()
    return {
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "labels_sha256": hashlib.sha256(
            np.ascontiguousarray(labels).tobytes()
        ).hexdigest(),
    }


class CachedIRQCountDataset(Dataset):
    """Same ((matrix,index),label) contract as IRQCountDataset.

    The float32 matrix and int64 index use read-only numpy memmap; each
    returned matrix is copied to a writable ndarray for safe torch collation.
    """

    def __init__(self, cache_dir, split, labels, source_npz, args):
        root = Path(cache_dir)
        meta_path = root / f"{split}.meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Run tools/precompute_wtcm.py first: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("feature_config") != feature_config(args):
            raise ValueError(
                f"IRQ-WTCM parameters differ from cache for {split}. "
                "Rebuild cache with the current model parameters."
            )
        self.labels = np.asarray(labels)
        if meta.get("source_info") != source_info(source_npz, self.labels):
            raise ValueError(
                f"Source NPZ or labels changed since {split} cache creation. "
                "Rebuild cache to avoid silently mixing data splits."
            )
        # Old cache metadata remains supported with strict size/mtime and sample
        # revalidation; newly created caches also bind the full NPZ SHA-256.
        if meta.get("source_sha256") is not None and meta["source_sha256"] != file_sha256(source_npz):
            raise ValueError(f"{split}: source NPZ content hash differs from cache. Regenerate cache.")
        self.features = np.load(root / f"{split}_matrix.npy", mmap_mode="r", allow_pickle=False)
        self.indices = np.load(root / f"{split}_index.npy", mmap_mode="r", allow_pickle=False)
        expected = (
            len(self.labels), 1,
            args.maximum_cell_number + 3, args.max_matrix_len
        )
        if self.features.shape != expected or self.features.dtype != np.float32:
            raise ValueError(f"Cache has shape/dtype {self.features.shape}/{self.features.dtype}, expected {expected}/float32")
        if self.indices.shape != (len(self.labels),):
            raise ValueError("Cache index count differs from label count")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        matrix = np.array(self.features[index], dtype=np.float32, copy=True)
        idx = int(self.indices[index])
        return (matrix, idx), self.labels[index]

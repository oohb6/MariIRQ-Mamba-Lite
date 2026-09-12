import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mariirq.mariirq_mamba_lite import MariIRQMambaLite
from mariirq.utils import evaluate_classification
from mariirq.wicm import WICMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MariIRQ-Mamba-Lite")
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/maritime"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/mariirq_mamba_lite/max_f1.pth")
    )
    parser.add_argument("--result_file", type=Path, default=Path("results/mariirq_mamba_lite.json"))

    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--wicm_window_size", type=int, default=20)
    parser.add_argument("--wicm_window_stride", type=int, default=20)
    parser.add_argument("--high_io_threshold", type=float, default=800.0)
    parser.add_argument("--log_transform", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MariIRQ-Mamba-Lite requires a CUDA GPU with the tested mamba-ssm setup")

    data = np.load(args.data_path / "test.npz")
    test_X, test_y = data["X"], data["y"]
    num_classes = len(np.unique(test_y))
    num_windows = (args.seq_len - args.wicm_window_size) // args.wicm_window_stride + 1

    dataset = WICMDataset(test_X, test_y, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = MariIRQMambaLite(
        num_classes=num_classes,
        num_rows=4,
        max_matrix_len=num_windows,
        embed_dim=args.embed_dim,
        depth=args.depth,
        headdim=args.headdim,
        drop_path_rate=args.drop_path_rate,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    model.to(device)

    metrics = evaluate_classification(model, loader, device)
    print(metrics)

    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    output = {
        **metrics,
        "metadata": {
            "samples": len(test_y),
            "classes": num_classes,
            "wicm_shape": [4, num_windows],
            "window_size": args.wicm_window_size,
            "high_io_threshold": args.high_io_threshold,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    }
    args.result_file.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mariirq.countmamba_backbone import MariIRQAdapt
from mariirq.irq_wtcm import IRQCountDataset
from mariirq.utils import evaluate_classification


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MariIRQ-Adapt")
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/maritime"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mariirq_adapt/max_f1.pth"))
    parser.add_argument("--result_file", type=Path, default=Path("results/mariirq_adapt.json"))

    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--maximum_load_time", type=float, default=5.0)
    parser.add_argument("--max_matrix_len", type=int, default=4998)
    parser.add_argument("--maximum_cell_number", type=int, default=32)
    parser.add_argument("--irq_bin_size", type=float, default=75.0)
    parser.add_argument("--irq_amp_threshold", type=float, default=None)
    parser.add_argument("--irq_cluster_window", type=int, default=20)
    parser.add_argument("--log_transform", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--drop_path_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The tested MariIRQ-Adapt configuration requires a CUDA GPU")

    data = np.load(args.data_path / "test.npz")
    test_X, test_y = data["X"], data["y"]
    num_classes = len(np.unique(test_y))

    dataset = IRQCountDataset(test_X, test_y, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = MariIRQAdapt(
        num_classes=num_classes,
        drop_path_rate=args.drop_path_rate,
        embed_dim=args.embed_dim,
        depth=args.depth,
        patch_size=args.maximum_cell_number + 3,
        max_matrix_len=args.max_matrix_len,
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
            "irq_bin_size": args.irq_bin_size,
            "irq_cluster_window": args.irq_cluster_window,
            "max_matrix_len": args.max_matrix_len,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    }
    args.result_file.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

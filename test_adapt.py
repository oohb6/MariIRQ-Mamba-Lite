import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mariirq.countmamba_backbone import MariIRQAdapt
from mariirq.irq_wtcm import IRQCountDataset
from mariirq.utils import evaluate_detailed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MariIRQ-Adapt")
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/maritime"))
    parser.add_argument("--test_file", type=Path, default=None, help="Override data_path/test.npz")
    parser.add_argument("--unknown_label", type=int, default=None, help="100 for 101-class supervised OW")
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

    data = np.load(args.test_file or args.data_path / "test.npz")
    test_X, test_y = data["X"], data["y"]
    # Always derive output dimension from TRAINED checkpoint, not observed TEST classes.
    state = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(state["fc.weight"].shape[0])
    if test_X.ndim != 3 or test_X.shape[1:] != (5000, 2):
        raise ValueError(f"Expected MariIRQ X=(N,5000,2), got {test_X.shape}. Use tools/prepare_baseline.py with RAW IRQ first.")
    if len(test_y) == 0 or test_y.min() < 0 or test_y.max() >= num_classes:
        raise ValueError(f"Test y range incompatible with frozen {num_classes}-class head; mapping cannot be re-indexed.")

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
    model.load_state_dict(state, strict=True)
    model.to(device)

    metrics = evaluate_detailed(model, loader, device, num_classes, unknown_label=args.unknown_label)
    print(metrics)

    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    output = {
        **metrics,
        "metadata": {
            "samples": len(test_y),
            "model_output_classes": num_classes,
            "observed_test_classes": int(len(np.unique(test_y))),
            "test_file": str(args.test_file or args.data_path / "test.npz"),
            "irq_bin_size": args.irq_bin_size,
            "irq_cluster_window": args.irq_cluster_window,
            "max_matrix_len": args.max_matrix_len,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    }
    args.result_file.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

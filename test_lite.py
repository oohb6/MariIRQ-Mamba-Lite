import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mariirq.mariirq_mamba_lite import MariIRQMambaLite
from mariirq.utils import evaluate_detailed
from mariirq.wicm import WICMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MariIRQ-Mamba-Lite")
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/maritime"))
    parser.add_argument("--test_file", type=Path, default=None, help="Override data_path/test.npz")
    parser.add_argument("--unknown_label", type=int, default=None, help="100 for 101-class supervised OW")
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

    data = np.load(args.test_file or args.data_path / "test.npz")
    test_X, test_y = data["X"], data["y"]
    # Always derive output dimension from TRAINED checkpoint, not observed TEST classes.
    state = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(state["fc.weight"].shape[0])
    if test_X.ndim != 3 or test_X.shape[1:] != (5000, 2):
        raise ValueError(f"Expected MariIRQ X=(N,5000,2), got {test_X.shape}. Use tools/prepare_baseline.py with RAW IRQ first.")
    if len(test_y) == 0 or test_y.min() < 0 or test_y.max() >= num_classes:
        raise ValueError(f"Test y range incompatible with frozen {num_classes}-class head; mapping cannot be re-indexed.")
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
    model.load_state_dict(state, strict=True)
    model.to(device)

    metrics = evaluate_detailed(model, loader, device, num_classes, unknown_label=args.unknown_label)
    print(metrics)

    # 导出逐样本预测；loader 不打乱，因此顺序与 test.npz 中的 y 一致。
    model.eval()
    true_parts, pred_parts = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = batch[0], batch[1]
            # WICMDataset yields ((matrix, last_idx), label); Lite only needs matrix.
            matrix = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            logits = model(matrix.to(device, non_blocking=True))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            predictions = logits.argmax(dim=1)
            true_parts.append(labels.cpu().numpy())
            pred_parts.append(predictions.cpu().numpy())

    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = args.result_file.with_name(
        args.result_file.stem + "_predictions.npz"
    )
    np.savez_compressed(
        prediction_file,
        y_true=np.concatenate(true_parts).astype(np.int64),
        y_pred=np.concatenate(pred_parts).astype(np.int64),
    )
    print(f"逐样本预测已保存：{prediction_file}")

    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    output = {
        **metrics,
        "metadata": {
            "samples": len(test_y),
            "model_output_classes": num_classes,
            "observed_test_classes": int(len(np.unique(test_y))),
            "test_file": str(args.test_file or args.data_path / "test.npz"),
            "wicm_shape": [4, num_windows],
            "window_size": args.wicm_window_size,
            "high_io_threshold": args.high_io_threshold,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    }
    args.result_file.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

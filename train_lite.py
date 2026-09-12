import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from mariirq.mariirq_mamba_lite import MariIRQMambaLite
from mariirq.utils import adjust_learning_rate, evaluate_classification
from mariirq.wicm import WICMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train MariIRQ-Mamba-Lite")
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/maritime"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/mariirq_mamba_lite"))

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
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_npz(path):
    data = np.load(path)
    return data["X"], data["y"]


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.embed_dim % args.headdim != 0:
        raise ValueError("embed_dim must be divisible by headdim")
    if not torch.cuda.is_available():
        raise RuntimeError("MariIRQ-Mamba-Lite requires a CUDA GPU with the tested mamba-ssm setup")

    train_X, train_y = load_npz(args.data_path / "train.npz")
    valid_X, valid_y = load_npz(args.data_path / "valid.npz")
    num_classes = len(np.unique(train_y))
    num_windows = (args.seq_len - args.wicm_window_size) // args.wicm_window_stride + 1

    train_set = WICMDataset(train_X, train_y, args)
    valid_set = WICMDataset(valid_X, valid_y, args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set,
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "max_f1.pth"
    best_f1 = -1.0

    print(f"Train/Valid: {len(train_set)}/{len(valid_set)} | Classes: {num_classes}")
    print(f"WICM: 4 x {num_windows} | window/stride={args.wicm_window_size}/{args.wicm_window_stride}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.6f} M")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            adjust_learning_rate(optimizer, epoch + step / len(train_loader), args)
            x = batch[0][0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            sample_count += x.size(0)

        metrics = evaluate_classification(model, valid_loader, device)
        print(
            f"Epoch {epoch + 1}: loss={running_loss / max(sample_count, 1):.5f}, "
            f"val={metrics}, lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        if metrics["F1-score"] > best_f1:
            best_f1 = metrics["F1-score"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved: {checkpoint_path}")

    print(f"Best validation F1: {best_f1:.2f}")


if __name__ == "__main__":
    main()

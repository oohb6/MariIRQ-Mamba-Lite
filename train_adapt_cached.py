import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from mariirq.countmamba_backbone import MariIRQAdapt
from mariirq.irq_wtcm import IRQCountDataset
from mariirq.irq_wtcm_cached import CachedIRQCountDataset
from mariirq.utils import adjust_learning_rate, evaluate_classification


def parse_args():
    parser = argparse.ArgumentParser(description="Train MariIRQ-Adapt")
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/maritime"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/mariirq_adapt"))

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
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--cache_dir", type=Path, default=None, help="Precomputed IRQ-WTCM cache directory")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10, help="Stop after this many consecutive epochs without a validation F1 improvement")
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
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"]


def load_labels(path):
    with np.load(path, allow_pickle=False) as data:
        return data["y"]


def main():
    args = parse_args()
    if args.patience < 1:
        raise ValueError("--patience must be >= 1")
    set_seed(args.seed)

    if args.max_matrix_len % 6 != 0:
        raise ValueError("max_matrix_len must be divisible by 6")
    if not torch.cuda.is_available():
        raise RuntimeError("The tested MariIRQ-Adapt configuration requires a CUDA GPU")

    if args.cache_dir is not None:
        # Features come from verified memmaps: read only y, not full trace X.
        train_y = load_labels(args.data_path / "train.npz")
        valid_y = load_labels(args.data_path / "valid.npz")
    else:
        train_X, train_y = load_npz(args.data_path / "train.npz")
        valid_X, valid_y = load_npz(args.data_path / "valid.npz")
    num_classes = len(np.unique(train_y))

    if args.cache_dir is not None:
        train_set = CachedIRQCountDataset(
            args.cache_dir, "train", train_y, args.data_path / "train.npz", args
        )
        valid_set = CachedIRQCountDataset(
            args.cache_dir, "valid", valid_y, args.data_path / "valid.npz", args
        )
        print(f"Using verified IRQ-WTCM cache: {args.cache_dir}")
    else:
        train_set = IRQCountDataset(train_X, train_y, args)
        valid_set = IRQCountDataset(valid_X, valid_y, args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    device = torch.device("cuda")
    model = MariIRQAdapt(
        num_classes=num_classes,
        drop_path_rate=args.drop_path_rate,
        embed_dim=args.embed_dim,
        depth=args.depth,
        patch_size=args.maximum_cell_number + 3,
        max_matrix_len=args.max_matrix_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "max_f1.pth"
    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    epochs_ran = 0

    print(f"Train/Valid: {len(train_set)}/{len(valid_set)} | Classes: {num_classes}")
    print(f"IRQ-WTCM: {args.maximum_cell_number + 3} x {args.max_matrix_len} | bin={args.irq_bin_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.6f} M")

    epoch_times = []
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        epochs_ran = epoch + 1
        model.train()
        running_loss = torch.zeros((), device=device)
        sample_count = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            adjust_learning_rate(optimizer, epoch + step / len(train_loader), args)
            x = batch[0][0].to(device, non_blocking=True)
            idx = batch[0][1].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, idx)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach() * x.size(0)
            sample_count += x.size(0)

        metrics = evaluate_classification(model, valid_loader, device)
        torch.cuda.synchronize()
        epoch_elapsed = time.perf_counter() - epoch_started
        epoch_times.append(round(epoch_elapsed, 3))
        print(
            f"Epoch {epoch + 1}: loss={running_loss.item() / max(sample_count, 1):.5f}, "
            f"val={metrics}, lr={optimizer.param_groups[0]['lr']:.6g}, time={epoch_elapsed:.1f}s"
        )

        current_f1 = metrics["F1-score"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved: {checkpoint_path} (best_epoch={best_epoch}, val_f1={best_f1:.2f})")
        else:
            no_improve += 1
            print(f"No val F1 improvement: {no_improve}/{args.patience} epochs; "
                  f"best={best_f1:.2f} at epoch {best_epoch}")

        if no_improve >= args.patience:
            print(f"EARLY STOP at epoch {epoch + 1}: validation F1 did not improve "
                  f"for {args.patience} consecutive epochs")
            break

    import json
    summary = {
        "seed": args.seed,
        "max_epochs": args.epochs,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "patience": args.patience,
        "stopped_early": epochs_ran < args.epochs,
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "epoch_times_sec": epoch_times,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best validation F1: {best_f1:.2f} (epoch {best_epoch}; trained {epochs_ran}/{args.epochs})")


if __name__ == "__main__":
    main()

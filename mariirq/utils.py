import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def measurement(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "F1-score": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
    }


def evaluate_classification(model, loader, device):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0][0].to(device, non_blocking=True)
            idx = batch[0][1].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            logits = model(x, idx)
            pred = logits.argmax(dim=1)
            predictions.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())

    return measurement(np.concatenate(labels), np.concatenate(predictions))


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sine-cosine positional embeddings")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / max(args.warmup_epochs, 1)
    else:
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

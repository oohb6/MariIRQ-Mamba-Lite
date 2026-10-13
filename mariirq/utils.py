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


def evaluate_detailed(model, loader, device, num_classes, unknown_label=None):
    """CW metrics incl Top-5; optionally supervised 101-class OW with class 100 unknown.

    Macro scores named Accuracy/Precision/Recall/F1-score keep the release's
    observed-label macro convention; fixed-class macro scores are separate.
    """
    from sklearn.metrics import precision_recall_fscore_support
    model.eval()
    ps=[]; ys=[]; top5_hits=0; n=0
    with torch.inference_mode():
        for batch in loader:
            x=batch[0][0].to(device,non_blocking=True)
            idx=batch[0][1].to(device,non_blocking=True)
            y=batch[1].to(device,non_blocking=True).long()
            logits=model(x,idx)
            pred=logits.argmax(1)
            k=min(5,logits.shape[1])
            top5_hits+=int((logits.topk(k,dim=1).indices==y.unsqueeze(1)).any(dim=1).sum().item())
            n+=len(y);ps.append(pred.cpu().numpy());ys.append(y.cpu().numpy())
    yp=np.concatenate(ps);yt=np.concatenate(ys)
    m=measurement(yt,yp)
    m['Top-5']=round(100.*top5_hits/n,2)
    m['Macro-F1 (all output classes)']=round(f1_score(yt,yp,average='macro',labels=np.arange(num_classes),zero_division=0)*100,2)
    if unknown_label is not None:
        if unknown_label<0 or unknown_label>=num_classes:raise ValueError('unknown label outside trained output classes')
        known=yt!=unknown_label;unknown=~known
        if not known.any() or not unknown.any():raise ValueError('101-class OW test requires both known and unknown samples')
        pred_unknown=yp==unknown_label
        from sklearn.metrics import precision_score as ps_,recall_score as rs_,f1_score as fs_
        m.update({
            'Sensitive Acc':round(100*np.mean(yp[known]==yt[known]),2),
            'Non-sensitive Acc':round(100*np.mean(yp[unknown]==unknown_label),2),
            'Combined Acc':m['Accuracy'],
            'Unknown FPR':round(100*np.mean(yp[unknown]!=unknown_label),2),
            'Unknown binary precision':round(100*ps_(unknown,pred_unknown,zero_division=0),2),
            'Unknown binary recall':round(100*rs_(unknown,pred_unknown,zero_division=0),2),
            'Unknown binary F1':round(100*fs_(unknown,pred_unknown,zero_division=0),2),
            'Known acceptance':round(100*np.mean(~pred_unknown[known]),2),
        })
    return m

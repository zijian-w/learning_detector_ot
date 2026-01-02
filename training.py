from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .model import CNNstack
except ImportError:  # allows running as a plain script/notebook with `ivml_cleanup/` on sys.path
    from model import CNNstack


def _relative_l2_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch = y.shape[0]
    diff = (yhat - y).reshape(batch, -1)
    y_flat = y.reshape(batch, -1)
    denom = torch.linalg.vector_norm(y_flat, dim=1).clamp_min(1e-12)
    return (torch.linalg.vector_norm(diff, dim=1) / denom).mean()


def train(
    x: np.ndarray,
    y: np.ndarray,
    save_dir: str | Path,
    *,
    batch_size: int = 16,
    lr: float = 1e-3,
    lr_half_life: int = 30,
    max_epochs: int = 500,
    kernel_size: int = 3,
    hidden: int = 8192,
    seed: int = 123,
    patience: int = 100,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1 (got {max_epochs})")

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda")

    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)

    n = len(x_t)
    if n == 0:
        raise ValueError("Empty dataset: len(x) == 0")

    train_end = int(0.7 * n)
    val_end = int(0.9 * n)
    if train_end == 0 or val_end <= train_end:
        raise ValueError(f"Not enough data for train/val split (n={n})")

    x_train, y_train = x_t[:train_end], y_t[:train_end]
    x_val, y_val = x_t[train_end:val_end], y_t[train_end:val_end]

    dl_train = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    dl_val = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CNNstack(kernel_size=kernel_size, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_path = save_dir / "checkpoint_best.pt"
    last_path = save_dir / "checkpoint_last.pt"

    best_val_epoch = 0
    best_val_loss = float("inf")
    best_train_epoch = 0
    best_train_loss = float("inf")

    def save_checkpoint(path: Path, *, epoch: int, train_loss: float, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "model_name": model.name(),
                "model_kwargs": {"kernel_size": kernel_size, "hidden": hidden},
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = _relative_l2_loss(yhat, yb)
            loss.backward()
            optimizer.step()

            bsz = xb.shape[0]
            train_loss_sum += loss.item() * bsz
            train_count += bsz

        train_loss = train_loss_sum / train_count
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch = epoch

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = model(xb)
                loss = _relative_l2_loss(yhat, yb)
                bsz = xb.shape[0]
                val_loss_sum += loss.item() * bsz
                val_count += bsz

        val_loss = val_loss_sum / val_count

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:04d} | lr={current_lr:.2e} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            save_checkpoint(best_path, epoch=epoch, train_loss=train_loss, val_loss=val_loss)

        if epoch - best_train_epoch >= lr_half_life and current_lr > 1e-8:
            for group in optimizer.param_groups:
                group["lr"] = current_lr * 0.5
            best_train_epoch = epoch

        if epoch - best_val_epoch >= patience:
            break

    save_checkpoint(last_path, epoch=epoch, train_loss=train_loss, val_loss=val_loss)
    return best_path


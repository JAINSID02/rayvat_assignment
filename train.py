"""
train.py
--------
Trains the LSTM text generator on the preprocessed Shakespeare sequences.

Includes:
    - Train / validation split (already done in preprocess.py)
    - Adam optimizer + CrossEntropyLoss (= softmax + categorical
      cross-entropy, computed in one numerically-stable step)
    - Early stopping on validation loss
    - Best-model checkpointing
"""

import pickle
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import LSTMTextGenerator

SEQUENCES_PATH = "checkpoints/sequences.npz"
VOCAB_PATH = "checkpoints/vocab.pkl"


def get_dataloaders(batch_size=128):
    data = np.load(SEQUENCES_PATH)
    train_ds = TensorDataset(
        torch.from_numpy(data["train_x"]).long(),
        torch.from_numpy(data["train_y"]).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(data["val_x"]).long(),
        torch.from_numpy(data["val_y"]).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)
    return total_loss / total_count


def train(
    embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2,
    batch_size=128, lr=1e-3, max_epochs=15, patience=3,
    checkpoint_path="checkpoints/best_model.pt", tag="default",
    resume_from=None,
):
    """resume_from: optional path to an existing checkpoint to warm-start from."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab["idx2word"])

    train_loader, val_loader = get_dataloaders(batch_size)

    model = LSTMTextGenerator(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout,
    ).to(device)

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[{tag}] Resumed weights from {resume_from}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] Model params: {n_params:,} | "
          f"embed_dim={embed_dim} hidden_dim={hidden_dim} num_layers={num_layers}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # softmax + NLL in one stable op

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        start = time.time()
        running_loss, running_count = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_count += x.size(0)

        train_loss = running_loss / running_count
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(f"[{tag}] Epoch {epoch:02d}/{max_epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_ppl={np.exp(val_loss):.2f} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Checkpointing: save whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size, "embed_dim": embed_dim,
                    "hidden_dim": hidden_dim, "num_layers": num_layers,
                    "dropout": dropout,
                },
            }, checkpoint_path)
            print(f"  -> New best val_loss {val_loss:.4f}. Checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"[{tag}] Early stopping triggered at epoch {epoch}.")
            break

    return history, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    train(
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, max_epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience,
        checkpoint_path="checkpoints/best_model.pt", tag="main",
        resume_from=args.resume_from,
    )

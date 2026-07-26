"""
preprocess.py
--------------
Data loading and preprocessing for the LSTM text generator.

Steps:
1. Load raw text file.
2. Lowercase everything and strip punctuation (keeping words intact).
3. Tokenize on whitespace into a word-level token stream.
4. Build a vocabulary (word <-> index maps), capping vocab size to the
   most frequent words for a tractable embedding/softmax layer, with an
   <unk> token for everything else.
5. Convert the whole corpus into a single integer array.
6. Slice the integer array into (input_sequence, next_token) pairs using
   a sliding window, then split into train/validation sets.

This module is imported by train.py and generate.py so the exact same
preprocessing logic is reused everywhere.
"""

import re
import string
import pickle
from collections import Counter

import numpy as np

# ------------------------- Config -------------------------
DATA_PATH = "data/shakespeare.txt"
VOCAB_SIZE = 8000          # cap vocabulary to top-N most frequent words
SEQ_LEN = 20               # number of tokens fed in before predicting the next
VAL_SPLIT = 0.1
VOCAB_PATH = "checkpoints/vocab.pkl"
SEQUENCES_PATH = "checkpoints/sequences.npz"

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


def load_and_clean_text(path: str) -> str:
    """Read the raw text file, lowercase it, and strip punctuation."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.lower()
    # Remove punctuation but keep newlines/spaces as word separators.
    # (translate table maps every punctuation char to None)
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    # Collapse all whitespace (including newlines) to single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Whitespace tokenization into a flat list of word tokens."""
    return text.split(" ")


def build_vocab(tokens: list, vocab_size: int):
    """Build word2idx / idx2word restricted to the `vocab_size` most common
    tokens. Reserve index 0 for <pad> and index 1 for <unk>."""
    counter = Counter(tokens)
    most_common = counter.most_common(vocab_size - 2)  # leave room for pad/unk

    idx2word = [PAD_TOKEN, UNK_TOKEN] + [w for w, _ in most_common]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def encode(tokens: list, word2idx: dict) -> np.ndarray:
    """Map each token to its integer id, falling back to <unk>."""
    unk_id = word2idx[UNK_TOKEN]
    return np.array([word2idx.get(t, unk_id) for t in tokens], dtype=np.int32)


def make_sequences(encoded: np.ndarray, seq_len: int):
    """Slide a window of length `seq_len` over the encoded corpus to build
    (input, target) pairs, where target = the token right after the window."""
    inputs, targets = [], []
    for i in range(len(encoded) - seq_len):
        inputs.append(encoded[i:i + seq_len])
        targets.append(encoded[i + seq_len])
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)


def run_pipeline():
    print(f"Loading raw text from {DATA_PATH} ...")
    text = load_and_clean_text(DATA_PATH)
    tokens = tokenize(text)
    print(f"Total tokens after cleaning: {len(tokens):,}")

    word2idx, idx2word = build_vocab(tokens, VOCAB_SIZE)
    print(f"Vocabulary size (incl. <pad>/<unk>): {len(idx2word):,}")

    encoded = encode(tokens, word2idx)
    inputs, targets = make_sequences(encoded, SEQ_LEN)
    print(f"Built {len(inputs):,} (input, target) training pairs "
          f"with sequence length {SEQ_LEN}.")

    # Shuffle then split into train / validation
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(inputs))
    inputs, targets = inputs[perm], targets[perm]

    n_val = int(len(inputs) * VAL_SPLIT)
    val_x, val_y = inputs[:n_val], targets[:n_val]
    train_x, train_y = inputs[n_val:], targets[n_val:]

    print(f"Train pairs: {len(train_x):,} | Val pairs: {len(val_x):,}")

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word,
                     "seq_len": SEQ_LEN}, f)

    np.savez_compressed(
        SEQUENCES_PATH,
        train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
    )
    print(f"Saved vocab -> {VOCAB_PATH}")
    print(f"Saved sequences -> {SEQUENCES_PATH}")


if __name__ == "__main__":
    run_pipeline()

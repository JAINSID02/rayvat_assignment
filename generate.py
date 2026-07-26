"""
generate.py
-----------
Loads a trained checkpoint and generates new text from a seed phrase by
iteratively predicting the next word and feeding it back into the model.

Sampling strategy: temperature-scaled multinomial sampling (rather than
pure argmax) so the output isn't stuck repeating the single most likely
word over and over -- a common failure mode of greedy decoding on small
language models.
"""

import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import LSTMTextGenerator
from preprocess import load_and_clean_text, tokenize

VOCAB_PATH = "checkpoints/vocab.pkl"
CHECKPOINT_PATH = "checkpoints/best_model.pt"


def load_model(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = LSTMTextGenerator(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def encode_seed(seed_text, word2idx, seq_len, unk_id):
    """Clean + tokenize the seed the same way training data was processed,
    then left-pad/truncate to exactly `seq_len` tokens."""
    cleaned = load_and_clean_text_from_string(seed_text)
    tokens = tokenize(cleaned)
    ids = [word2idx.get(t, unk_id) for t in tokens]
    if len(ids) < seq_len:
        ids = [word2idx["<pad>"]] * (seq_len - len(ids)) + ids
    else:
        ids = ids[-seq_len:]
    return ids


def load_and_clean_text_from_string(text):
    """Same cleaning logic as preprocess.load_and_clean_text but operating
    on an in-memory string instead of a file path."""
    import re
    import string as string_mod
    text = text.lower()
    translator = str.maketrans("", "", string_mod.punctuation)
    text = text.translate(translator)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def generate_text(model, seed_text, word2idx, idx2word, seq_len,
                   num_words=50, temperature=0.8, device="cpu"):
    unk_id = word2idx["<unk>"]
    ids = encode_seed(seed_text, word2idx, seq_len, unk_id)
    generated = list(ids)

    output_words = []
    for _ in range(num_words):
        window = torch.tensor([generated[-seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(window)
        logits = logits.squeeze(0) / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        output_words.append(idx2word[next_id])

    return " ".join(output_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="first citizen before we proceed")
    parser.add_argument("--num_words", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    word2idx, idx2word, seq_len = vocab["word2idx"], vocab["idx2word"], vocab["seq_len"]

    model = load_model(args.checkpoint, device=device)

    result = generate_text(
        model, args.seed, word2idx, idx2word, seq_len,
        num_words=args.num_words, temperature=args.temperature, device=device,
    )
    print(f"\nSEED: {args.seed}")
    print(f"GENERATED ({args.temperature=}):\n{args.seed} {result}\n")


if __name__ == "__main__":
    main()

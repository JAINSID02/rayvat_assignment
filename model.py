"""
model.py
--------
LSTM-based next-word language model.

Architecture:
    Embedding  -> stacked LSTM layers -> Dropout -> Linear (vocab logits)

The final Linear layer outputs raw logits over the vocabulary; softmax is
applied implicitly by nn.CrossEntropyLoss during training (and explicitly
with a temperature at generation time).
"""

import torch
import torch.nn as nn


class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # dense output layer (logits)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) token ids
        returns: logits (batch, vocab_size) for the NEXT token,
                 taken from the last LSTM timestep.
        """
        emb = self.embedding(x)                        # (batch, seq_len, embed_dim)
        lstm_out, hidden = self.lstm(emb, hidden)        # (batch, seq_len, hidden_dim)
        last_step = lstm_out[:, -1, :]                   # (batch, hidden_dim)
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)                      # (batch, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        return (h0, c0)

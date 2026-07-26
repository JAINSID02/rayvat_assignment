# LSTM Text Generator — Shakespeare

A word-level LSTM language model trained on Shakespeare's plays that
generates new text from a seed phrase.

## 1. Dataset

**Tiny Shakespeare** — a ~1.1MB public-domain text corpus of Shakespeare's
plays, originally compiled by Andrej Karpathy for the `char-rnn` project.

- Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Already included in this repo at `data/shakespeare.txt`.
- To re-download it:
  ```bash
  curl -sL -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  ```

## 2. Project structure

```
RAYVAT_ASSIGNMENT/
├── data/
│   └── shakespeare.txt        # raw corpus
├── checkpoints/
│   ├── vocab.pkl              # word2idx / idx2word + seq_len (created by preprocess.py)
│   ├── sequences.npz          # preprocessed train/val pairs (created by preprocess.py)
│   └── best_model.pt          # trained model weights (created by train.py)
├── preprocess.py               # Data preprocessing
├── model.py                     # LSTM model architecture
├── train.py                     # Training loop, early stopping, checkpointing
├── generate.py                  # Seed -> generated text
└── README.md
```

## 3. How to run

```bash
pip install torch numpy

python preprocess.py                 # builds checkpoints/vocab.pkl + sequences.npz
python train.py --epochs 8           # trains and saves checkpoints/best_model.pt
python generate.py --seed "to be or not to be" --num_words 40 --temperature 0.75
```

## 4. Approach

### 4.1 Preprocessing (`preprocess.py`)
- Lowercases the full corpus and strips all punctuation.
- Whitespace-tokenizes into word tokens.
- Builds a vocabulary capped at the most frequent words (plus `<pad>`/`<unk>`
  for padding and rare/unseen words), keeping the embedding and output
  layers a manageable size.
- Slides a fixed-length window over the encoded corpus to build
  (input sequence, next-word target) training pairs, then splits them
  into train/validation sets.

### 4.2 Model (`model.py`)
`Embedding → LSTM → Dropout → Linear`

The final layer outputs raw logits over the vocabulary. Softmax is
applied implicitly during training via `nn.CrossEntropyLoss` (softmax +
cross-entropy combined in one numerically stable step), and explicitly
with a temperature during text generation.

### 4.3 Training (`train.py`)
- Adam optimizer with gradient clipping.
- **Checkpointing:** saves the model whenever validation loss improves.
- **Early stopping:** halts training if validation loss doesn't improve
  for a set number of epochs, to avoid overfitting.
- Prints training/validation loss and perplexity after every epoch.

### 4.4 Generation (`generate.py`)
- Cleans and tokenizes the seed phrase using the same pipeline as
  training data.
- Generates text **iteratively**: predicts the next word, appends it,
  slides the window forward, and repeats.
- Uses **temperature-scaled sampling** instead of greedy decoding, so
  output doesn't collapse into repeating the single most likely word.

## 5. Sample Generated Outputs


**Seed:** `"first citizen before we proceed"` (temperature=0.75)
> first citizen before we proceed more than but i <unk> my gentle true i have we will be a contract to an part and to the tower and <unk> all the goodness and that we had he will make you who had rather the <unk>

**Seed:** `"o romeo wherefore art thou"` (temperature=0.8)
> o romeo wherefore art thou shalt make hard so our general be <unk> and that that is it is my heart and most birth in all the golden voice of twelve are a foreign age that which i am sure such our <unk> did i

**Seed:** `"the king said unto him"` (temperature=0.7)
> the king said unto him with it as a <unk> of any thing of breath as we shall be so much itself for the king and to the bush and his <unk> which is gone and eat my father let him welcome o in my house and being a thousand word in this <unk> and

**Note on quality:** the model captures Shakespeare's archaic vocabulary and
short clause rhythm, but doesn't maintain long-range coherence — this is
expected given the small model size and limited training epochs. Training
longer, using more data, or switching to character-level tokenization would
likely improve coherence further.
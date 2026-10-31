import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA = "data/processed_liar2_v1"

ORIGINAL = "models/fdhn_liar2_regularized/best_model.pt"
FROZEN = "models/frozen_roberta_fdhn_liar2/best_model.pt"

LABELS = ["pants-fire", "false", "barely-true",
          "half-true", "mostly-true", "true"]

train = pd.read_csv(f"{DATA}/train.csv").fillna("")
valid = pd.read_csv(f"{DATA}/validation.csv").fillna("")

original_ckpt = torch.load(ORIGINAL, map_location=DEVICE)
frozen_ckpt = torch.load(FROZEN, map_location=DEVICE)

META = original_ckpt["meta_columns"]
NUM = original_ckpt["history_columns"]

verdict_re = re.compile(
    r"\bpants?\s+on\s+fire\b|\bbarely[\s-]+true\b|\bhalf[\s-]+true\b|"
    r"\bmostly[\s-]+true\b|\bmostly[\s-]+false\b|\bmostly[\s-]+untrue\b|"
    r"\btrue\b|\bfalse\b|\buntrue\b|\btruthful\b|\btruthfulness\b", re.I
)

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def clean_just(x):
    return re.sub(r"\s+", " ", verdict_re.sub(" ", clean(x))).strip()

def meta_text(row):
    return " [SEP] ".join(f"{c}: {clean(row[c])}" for c in META)

def encode(text, vocab, length):
    ids = [vocab.get(w, 1) for w in
           re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())[:length]]
    return ids + [0] * (length - len(ids))

def prepare_meta(df, vocab, length):
    texts = df.apply(meta_text, axis=1).tolist()
    return np.asarray([encode(x, vocab, length) for x in texts],
                      dtype=np.int64)

def prepare_nums(df, checkpoint):
    values = df[NUM].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).to_numpy(dtype=np.float32)

    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    return ((values - mean) / scale).astype(np.float32)

# ---------------- MODEL DEFINITIONS ----------------

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=128, filters=128, drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb, filters, k) for k in (3, 4, 5)
        ])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(filters * 3, 6)

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            z = F.relu(conv(x))
            pooled.append(F.max_pool1d(z, z.size(2)).squeeze(2))
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 64)
        self.conv = nn.Conv1d(64, 32, 1)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(
            self.conv(self.proj(x.unsqueeze(-1)).transpose(1, 2))
        ).transpose(1, 2)
        seq, _ = self.lstm(x)
        return self.fc(self.drop(seq[:, -1, :]))


class Fuzzy(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(6, 6))
        self.raw_sigma = nn.Parameter(torch.zeros(6, 6))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x):
        sigma = F.softplus(self.raw_sigma) + 1e-4
        return torch.exp(
            -0.5 * (
                (x.unsqueeze(1) - self.centers.unsqueeze(0))
                / sigma.unsqueeze(0)
            ) ** 2
        ).mean(-1)


class OriginalFDHN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.meta = TextCNN(vocab_size)
        self.just = TextCNN(vocab_size)
        self.numeric = CNNBiLSTM()
        self.fuzzy = Fuzzy()
        self.drop = nn.Dropout(0.3)
        self.fusion = nn.Linear(24, 6)

    def forward(self, m, j, n):
        a = self.meta(m)
        b = self.just(j)
        c = self.numeric(n)
        d = self.fuzzy(c)
        return self.fusion(self.drop(torch.cat([a, b, c, d], dim=1)))


class FrozenRoBERTaFDHN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )
        self.meta = TextCNN(vocab_size)
        self.numeric = CNNBiLSTM()
        self.fuzzy = Fuzzy()
        self.drop = nn.Dropout(0.3)
        self.fusion = nn.Linear(24, 6)

    def forward(self, meta_ids, text_embedding, nums):
        a = self.text_projection(text_embedding)
        b = self.meta(meta_ids)
        c = self.numeric(nums)
        d = self.fuzzy(c)
        return self.fusion(self.drop(torch.cat([a, b, c, d], dim=1)))


# ---------------- ORIGINAL FDHN ----------------

orig_model = OriginalFDHN(len(original_ckpt["vocab"])).to(DEVICE)
orig_model.load_state_dict(original_ckpt["model_state_dict"])
orig_model.eval()

om = prepare_meta(
    valid, original_ckpt["vocab"], original_ckpt["max_len_meta"]
)

just_text = valid.justification.map(clean_just).tolist()
LJ = original_ckpt["max_len_justification"]
oj = np.asarray([
    encode(x, original_ckpt["vocab"], LJ) for x in just_text
], dtype=np.int64)

on = prepare_nums(valid, original_ckpt)

# ---------------- FROZEN ROBERTA FDHN ----------------

frozen_model = FrozenRoBERTaFDHN(len(frozen_ckpt["vocab"])).to(DEVICE)
frozen_model.load_state_dict(frozen_ckpt["model_state_dict"])
frozen_model.eval()

fm = prepare_meta(
    valid, frozen_ckpt["vocab"], 100
)

fn = prepare_nums(valid, frozen_ckpt)

cached = np.load(
    "data/frozen_roberta_embeddings/validation_justification_roberta.npy"
)

if cached.shape != (len(valid), 768):
    raise ValueError(f"Unexpected embedding shape: {cached.shape}")

# ---------------- PREDICTIONS ----------------

class OriginalDataset(Dataset):
    def __init__(self):
        self.m = torch.tensor(om, dtype=torch.long)
        self.j = torch.tensor(oj, dtype=torch.long)
        self.n = torch.tensor(on, dtype=torch.float32)

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        return self.m[i], self.j[i], self.n[i]


class FrozenDataset(Dataset):
    def __init__(self):
        self.m = torch.tensor(fm, dtype=torch.long)
        self.e = torch.tensor(cached, dtype=torch.float32)
        self.n = torch.tensor(fn, dtype=torch.float32)

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        return self.m[i], self.e[i], self.n[i]


original_probs = []
frozen_probs = []

with torch.no_grad():
    for m, j, n in DataLoader(OriginalDataset(), batch_size=64):
        logits = orig_model(
            m.to(DEVICE), j.to(DEVICE), n.to(DEVICE)
        )
        original_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    for m, e, n in DataLoader(FrozenDataset(), batch_size=64):
        logits = frozen_model(
            m.to(DEVICE), e.to(DEVICE), n.to(DEVICE)
        )
        frozen_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

original_probs = np.concatenate(original_probs)
frozen_probs = np.concatenate(frozen_probs)

y = valid.label.map({v: i for i, v in enumerate(LABELS)}).to_numpy()

print("\nIndividual model results:")
for name, probs in [
    ("Original FDHN", original_probs),
    ("Frozen RoBERTa-FDHN", frozen_probs)
]:
    pred = probs.argmax(axis=1)
    print(
        f"{name}: accuracy={accuracy_score(y, pred)*100:.2f}% | "
        f"macro-F1={f1_score(y, pred, average='macro'):.4f}"
    )

print("\nTesting ensemble weights:")
results = []

for original_weight in np.arange(0.0, 1.01, 0.05):
    ensemble = (
        original_weight * original_probs
        + (1.0 - original_weight) * frozen_probs
    )
    pred = ensemble.argmax(axis=1)
    acc = accuracy_score(y, pred)
    mf1 = f1_score(y, pred, average="macro")

    results.append((original_weight, acc, mf1))
    print(
        f"Original={original_weight:.2f}, "
        f"Frozen={1-original_weight:.2f} | "
        f"Accuracy={acc*100:.2f}% | Macro-F1={mf1:.4f}"
    )

best = max(results, key=lambda x: (x[1], x[2]))

print("\n===== BEST VALIDATION ENSEMBLE =====")
print(f"Original FDHN weight: {best[0]:.2f}")
print(f"Frozen RoBERTa weight: {1-best[0]:.2f}")
print(f"Validation accuracy: {best[1]*100:.2f}%")
print(f"Macro-F1: {best[2]:.4f}")

pd.DataFrame(
    results, columns=["original_weight", "accuracy", "macro_f1"]
).to_csv("results/fdhn_ensemble_weight_comparison.csv", index=False)

print("Saved weight comparison: results/fdhn_ensemble_weight_comparison.csv")
print("Validation set only; test split was not loaded.")

import os, re, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
DATA = "data/processed_liar2_v1"
MODEL_NAME = "FacebookAI/roberta-base"

OUT = "models/frozen_roberta_fdhn_liar2"
REPORT = "results/frozen_roberta_fdhn_liar2_validation_report.csv"
CACHE = "data/frozen_roberta_embeddings"

os.makedirs(OUT, exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", DEVICE)

LABELS = ["pants-fire", "false", "barely-true",
          "half-true", "mostly-true", "true"]
label_id = {v: i for i, v in enumerate(LABELS)}

META = ["date", "subject", "speaker", "speaker_description",
        "state_info", "context"]

NUM = ["true_counts", "mostly_true_counts", "half_true_counts",
       "mostly_false_counts", "false_counts", "pants_on_fire_counts"]

# ---------------- DATA ----------------
train = pd.read_csv(f"{DATA}/train.csv").fillna("")
valid = pd.read_csv(f"{DATA}/validation.csv").fillna("")

for name, df in [("train", train), ("validation", valid)]:
    required = ["label", "justification"] + META + NUM
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    bad = set(df.label.astype(str)) - set(LABELS)
    if bad:
        raise ValueError(f"Unexpected labels in {name}: {bad}")

# Remove explicit verdict words from justification.
verdict_re = re.compile(
    r"\bpants?\s+on\s+fire\b|\bbarely[\s-]+true\b|\bhalf[\s-]+true\b|"
    r"\bmostly[\s-]+true\b|\bmostly[\s-]+false\b|\bmostly[\s-]+untrue\b|"
    r"\btrue\b|\bfalse\b|\buntrue\b|\btruthful\b|\btruthfulness\b",
    re.I
)

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def clean_just(x):
    return re.sub(r"\s+", " ", verdict_re.sub(" ", clean(x))).strip()

def meta_text(row):
    return " [SEP] ".join(f"{c}: {clean(row[c])}" for c in META)

tr_meta = train.apply(meta_text, axis=1).tolist()
va_meta = valid.apply(meta_text, axis=1).tolist()
tr_just = train.justification.map(clean_just).tolist()
va_just = valid.justification.map(clean_just).tolist()

# ---------------- METADATA VOCABULARY ----------------
tok_re = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
counter = Counter()

for s in tr_meta:
    counter.update(tok_re.findall(s.lower()))

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(29998):
    vocab[word] = len(vocab)

def encode_meta(s, max_len=100):
    ids = [vocab.get(w, 1) for w in tok_re.findall(s.lower())[:max_len]]
    return ids + [0] * (max_len - len(ids))

Xtm = np.asarray([encode_meta(s) for s in tr_meta], dtype=np.int64)
Xvm = np.asarray([encode_meta(s) for s in va_meta], dtype=np.int64)

# ---------------- NUMERICAL FEATURES ----------------
Xtn = train[NUM].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
Xvn = valid[NUM].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)

scaler = StandardScaler()
Xtn = scaler.fit_transform(Xtn).astype(np.float32)
Xvn = scaler.transform(Xvn).astype(np.float32)

yt = train.label.map(label_id).to_numpy(dtype=np.int64)
yv = valid.label.map(label_id).to_numpy(dtype=np.int64)

# ---------------- FROZEN ROBERTA EMBEDDINGS ----------------
print("Loading RoBERTa tokenizer and encoder...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, local_files_only=True
)
roberta = AutoModel.from_pretrained(
    MODEL_NAME, local_files_only=True, add_pooling_layer=False
).to(DEVICE)

roberta.eval()
for parameter in roberta.parameters():
    parameter.requires_grad = False

MAX_LEN = 256
EMBED_BATCH = 8

def embedding_cache_name(split):
    return Path(CACHE) / f"{split}_justification_roberta.npy"

def extract_embeddings(texts, split):
    cache_file = embedding_cache_name(split)

    if cache_file.exists():
        cached = np.load(cache_file)
        if cached.shape == (len(texts), 768):
            print(f"Using cached {split} embeddings:", cached.shape)
            return cached.astype(np.float32)

    print(f"Extracting {split} RoBERTa embeddings...")

    all_embeddings = []

    for start in range(0, len(texts), EMBED_BATCH):
        batch_texts = texts[start:start + EMBED_BATCH]

        encoded = tokenizer(
            batch_texts,
            max_length=MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = roberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

            # Masked mean pooling, excluding padding and special tokens.
            special = tokenizer.get_special_tokens_mask(
                input_ids[0].cpu().tolist(),
                already_has_special_tokens=True
            )
            special_mask = torch.tensor(
                [special], dtype=torch.bool, device=DEVICE
            ).expand(input_ids.size(0), -1)

            for row in range(1, input_ids.size(0)):
                row_special = tokenizer.get_special_tokens_mask(
                    input_ids[row].cpu().tolist(),
                    already_has_special_tokens=True
                )
                special_mask[row] = torch.tensor(
                    row_special, dtype=torch.bool, device=DEVICE
                )

            valid_mask = mask * (~special_mask).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * valid_mask).sum(dim=1)
            denominator = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled / denominator

        all_embeddings.append(pooled.cpu().numpy())

        if (start // EMBED_BATCH + 1) % 100 == 0:
            print(f"{split}: processed {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")

    result = np.concatenate(all_embeddings).astype(np.float32)
    np.save(cache_file, result)
    print(f"Saved {split} embeddings:", result.shape)
    return result

Xtr = extract_embeddings(tr_just, "train")
Xva = extract_embeddings(va_just, "validation")

del roberta
if DEVICE.type == "mps":
    torch.mps.empty_cache()

# ---------------- DATASET ----------------
class LIAR2Dataset(Dataset):
    def __init__(self, meta, text, nums, labels):
        self.meta = torch.tensor(meta, dtype=torch.long)
        self.text = torch.tensor(text, dtype=torch.float32)
        self.nums = torch.tensor(nums, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.meta[i], self.text[i], self.nums[i], self.labels[i]

BATCH_SIZE = 64

train_dl = DataLoader(
    LIAR2Dataset(Xtm, Xtr, Xtn, yt),
    batch_size=BATCH_SIZE, shuffle=True
)

valid_dl = DataLoader(
    LIAR2Dataset(Xvm, Xva, Xvn, yv),
    batch_size=BATCH_SIZE, shuffle=False
)

# ---------------- FDHN BRANCHES ----------------
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


class NumericCNNBiLSTM(nn.Module):
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
        membership = torch.exp(
            -0.5 * (
                (x.unsqueeze(1) - self.centers.unsqueeze(0))
                / sigma.unsqueeze(0)
            ) ** 2
        )
        return membership.mean(dim=-1)


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
        self.numeric = NumericCNNBiLSTM()
        self.fuzzy = Fuzzy()
        self.drop = nn.Dropout(0.3)
        self.fusion = nn.Linear(24, 6)

    def forward(self, meta_ids, text_embedding, nums):
        a = self.text_projection(text_embedding)
        b = self.meta(meta_ids)
        c = self.numeric(nums)
        d = self.fuzzy(c)
        return self.fusion(self.drop(torch.cat([a, b, c, d], dim=1)))


model = FrozenRoBERTaFDHN(len(vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0003, weight_decay=0.001
)

# ---------------- TRAINING ----------------
EPOCHS = 30
PATIENCE = 6
best_acc = -1
best_f1 = -1
best_epoch = 0
wait = 0
best_path = f"{OUT}/best_model.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    losses, train_preds, train_true = [], [], []

    for meta_ids, text_emb, nums, labels in train_dl:
        meta_ids = meta_ids.to(DEVICE)
        text_emb = text_emb.to(DEVICE)
        nums = nums.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(meta_ids, text_emb, nums)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        train_preds.extend(logits.argmax(1).detach().cpu().tolist())
        train_true.extend(labels.detach().cpu().tolist())

    model.eval()
    val_losses, val_preds, val_true = [], [], []

    with torch.no_grad():
        for meta_ids, text_emb, nums, labels in valid_dl:
            logits = model(
                meta_ids.to(DEVICE),
                text_emb.to(DEVICE),
                nums.to(DEVICE)
            )
            loss = criterion(logits, labels.to(DEVICE))
            val_losses.append(loss.item())
            val_preds.extend(logits.argmax(1).cpu().tolist())
            val_true.extend(labels.tolist())

    train_acc = accuracy_score(train_true, train_preds)
    val_acc = accuracy_score(val_true, val_preds)
    macro_f1 = f1_score(val_true, val_preds, average="macro")

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={np.mean(losses):.4f} "
        f"train_acc={train_acc*100:.2f}% | "
        f"val_loss={np.mean(val_losses):.4f} "
        f"val_acc={val_acc*100:.2f}% "
        f"val_macro_f1={macro_f1:.4f}",
        flush=True
    )

    if val_acc > best_acc or (
        np.isclose(val_acc, best_acc) and macro_f1 > best_f1
    ):
        best_acc, best_f1, best_epoch = val_acc, macro_f1, epoch
        wait = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "labels": LABELS,
            "meta_columns": META,
            "history_columns": NUM,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "best_epoch": best_epoch,
            "validation_accuracy": best_acc,
            "validation_macro_f1": best_f1,
            "embedding_model": MODEL_NAME
        }, best_path)
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.", flush=True)
            break

# ---------------- FINAL VALIDATION REPORT ----------------
checkpoint = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

final_true, final_preds = [], []

with torch.no_grad():
    for meta_ids, text_emb, nums, labels in valid_dl:
        logits = model(
            meta_ids.to(DEVICE),
            text_emb.to(DEVICE),
            nums.to(DEVICE)
        )
        final_preds.extend(logits.argmax(1).cpu().tolist())
        final_true.extend(labels.tolist())

acc = accuracy_score(final_true, final_preds)
mf1 = f1_score(final_true, final_preds, average="macro")
wf1 = f1_score(final_true, final_preds, average="weighted")

print("\n===== BEST FROZEN ROBERTA-FDHN CHECKPOINT =====")
print("Best epoch:", checkpoint["best_epoch"])
print(f"Validation accuracy: {acc*100:.2f}%")
print(f"Macro-F1: {mf1:.4f} | Weighted-F1: {wf1:.4f}")
print(classification_report(
    final_true, final_preds,
    labels=list(range(6)),
    target_names=LABELS,
    zero_division=0
))

report = classification_report(
    final_true, final_preds,
    labels=list(range(6)),
    target_names=LABELS,
    output_dict=True,
    zero_division=0
)
pd.DataFrame(report).transpose().to_csv(REPORT)

print("Saved model:", best_path)
print("Saved report:", REPORT)
print("Cached embeddings:", CACHE)
print("LIAR2 train/validation only; test split was not loaded.")

import os, re, random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# LIAR2-only FDHN experiment: metadata TextCNN + justification TextCNN +
# numerical CNN-BiLSTM + Gaussian fuzzy layer. Statement branch omitted.
SEED = 42
DATA = "data/processed_liar2_v1"
OUT = "models/fdhn_liar2_regularized"
REPORT = "results/fdhn_liar2_regularized_validation_report.csv"
os.makedirs(OUT, exist_ok=True)
os.makedirs("results", exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", DEVICE)

train = pd.read_csv(f"{DATA}/train.csv").fillna("")
valid = pd.read_csv(f"{DATA}/validation.csv").fillna("")
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
label_id = {v:i for i,v in enumerate(LABELS)}
META = ["date","subject","speaker","speaker_description","state_info","context"]
NUM = ["true_counts","mostly_true_counts","half_true_counts","mostly_false_counts","false_counts","pants_on_fire_counts"]
for name, df in [("train",train),("validation",valid)]:
    need = ["label","justification"] + META + NUM
    missing = [c for c in need if c not in df.columns]
    if missing: raise ValueError(f"{name} missing columns: {missing}")
    bad = set(df.label.astype(str)) - set(LABELS)
    if bad: raise ValueError(f"Unexpected {name} labels: {bad}")

# Conservative removal of explicit verdict words from justifications.
verdict_re = re.compile(
    r"\bpants?\s+on\s+fire\b|\bbarely[\s-]+true\b|\bhalf[\s-]+true\b|"
    r"\bmostly[\s-]+true\b|\bmostly[\s-]+false\b|\bmostly[\s-]+untrue\b|"
    r"\btrue\b|\bfalse\b|\buntrue\b|\btruthful\b|\btruthfulness\b", re.I)
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

tok_re = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
counter = Counter()
for s in tr_meta + tr_just: counter.update(tok_re.findall(s.lower()))
vocab = {"<PAD>":0, "<UNK>":1}
for word, _ in counter.most_common(29998): vocab[word] = len(vocab)
def encode(s, n):
    ids = [vocab.get(w,1) for w in tok_re.findall(s.lower())[:n]]
    return ids + [0]*(n-len(ids))
LM, LJ = 100, 180
Xtm = np.asarray([encode(s,LM) for s in tr_meta],dtype=np.int64)
Xvm = np.asarray([encode(s,LM) for s in va_meta],dtype=np.int64)
Xtj = np.asarray([encode(s,LJ) for s in tr_just],dtype=np.int64)
Xvj = np.asarray([encode(s,LJ) for s in va_just],dtype=np.int64)
Xtn = train[NUM].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=np.float32)
Xvn = valid[NUM].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=np.float32)
scaler = StandardScaler()
Xtn = scaler.fit_transform(Xtn).astype(np.float32)
Xvn = scaler.transform(Xvn).astype(np.float32)
yt = train.label.map(label_id).to_numpy(dtype=np.int64)
yv = valid.label.map(label_id).to_numpy(dtype=np.int64)
if pd.isna(train.label.map(label_id)).any() or pd.isna(valid.label.map(label_id)).any():
    raise ValueError("Unrecognized label.")

class DS(Dataset):
    def __init__(self,m,j,n,y):
        self.m=torch.tensor(m,dtype=torch.long); self.j=torch.tensor(j,dtype=torch.long)
        self.n=torch.tensor(n,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.m[i],self.j[i],self.n[i],self.y[i]
train_dl=DataLoader(DS(Xtm,Xtj,Xtn,yt),batch_size=64,shuffle=True)
valid_dl=DataLoader(DS(Xvm,Xvj,Xvn,yv),batch_size=64,shuffle=False)

class TextCNN(nn.Module):
    def __init__(self,vsize,emb=128,filters=128,drop=.5):
        super().__init__()
        self.emb=nn.Embedding(vsize,emb,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(emb,filters,k) for k in (3,4,5)])
        self.drop=nn.Dropout(drop); self.fc=nn.Linear(filters*3,6)
    def forward(self,x):
        x=self.emb(x).transpose(1,2); pooled=[]
        for conv in self.convs:
            z=F.relu(conv(x)); pooled.append(F.max_pool1d(z,z.size(2)).squeeze(2))
        return self.fc(self.drop(torch.cat(pooled,dim=1)))

class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(1,64); self.conv=nn.Conv1d(64,32,1)
        self.lstm=nn.LSTM(32,64,batch_first=True,bidirectional=True)
        self.drop=nn.Dropout(.3); self.fc=nn.Linear(128,6)
    def forward(self,x):
        x=F.relu(self.conv(self.proj(x.unsqueeze(-1)).transpose(1,2))).transpose(1,2)
        seq,_=self.lstm(x)
        return self.fc(self.drop(seq[:,-1,:]))

class Fuzzy(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers=nn.Parameter(torch.empty(6,6))
        self.raw_sigma=nn.Parameter(torch.zeros(6,6))
        nn.init.xavier_uniform_(self.centers)
    def forward(self,x):
        sig=F.softplus(self.raw_sigma)+1e-4
        return torch.exp(-.5*((x.unsqueeze(1)-self.centers.unsqueeze(0))/sig.unsqueeze(0))**2).mean(-1)

class FDHN(nn.Module):
    def __init__(self,vsize):
        super().__init__()
        self.meta=TextCNN(vsize); self.just=TextCNN(vsize)
        self.numeric=CNNBiLSTM(); self.fuzzy=Fuzzy()
        self.drop=nn.Dropout(.3); self.fusion=nn.Linear(24,6)
    def forward(self,m,j,n):
        a=self.meta(m); b=self.just(j); c=self.numeric(n); d=self.fuzzy(c)
        return self.fusion(self.drop(torch.cat([a,b,c,d],dim=1)))

model=FDHN(len(vocab)).to(DEVICE)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.001
)

EPOCHS = 20
PATIENCE = 5
best_acc=-1; best_f1=-1; best_epoch=0; wait=0
best_path=f"{OUT}/best_model.pt"

for epoch in range(1,EPOCHS+1):
    model.train(); losses=[]; p=[]; t=[]
    for m,j,n,y in train_dl:
        m,j,n,y=m.to(DEVICE),j.to(DEVICE),n.to(DEVICE),y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True); logits=model(m,j,n); loss=criterion(logits,y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        losses.append(loss.item()); p.extend(logits.argmax(1).detach().cpu().tolist()); t.extend(y.cpu().tolist())
    model.eval(); vl=[]; vp=[]; vt=[]
    with torch.no_grad():
        for m,j,n,y in valid_dl:
            logits=model(m.to(DEVICE),j.to(DEVICE),n.to(DEVICE))
            vl.append(criterion(logits,y.to(DEVICE)).item())
            vp.extend(logits.argmax(1).cpu().tolist()); vt.extend(y.tolist())
    acc=accuracy_score(vt,vp); mf1=f1_score(vt,vp,average="macro")
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={np.mean(losses):.4f} train_acc={accuracy_score(t,p)*100:.2f}% | val_loss={np.mean(vl):.4f} val_acc={acc*100:.2f}% val_macro_f1={mf1:.4f}",flush=True)
    if acc>best_acc or (np.isclose(acc,best_acc) and mf1>best_f1):
        best_acc,best_f1,best_epoch=acc,mf1,epoch; wait=0
        torch.save({"model_state_dict":model.state_dict(),"vocab":vocab,"labels":LABELS,
                    "meta_columns":META,"history_columns":NUM,"scaler_mean":scaler.mean_,
                    "scaler_scale":scaler.scale_,"max_len_meta":LM,"max_len_justification":LJ,
                    "best_epoch":best_epoch,"validation_accuracy":best_acc,
                    "validation_macro_f1":best_f1,"statement_branch_used":False,
                    "justification_verdict_scrubbing":True},best_path)
    else:
        wait+=1
        if wait>=PATIENCE: print("Early stopping.",flush=True); break

ckpt=torch.load(best_path,map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"]); model.eval(); vt=[]; vp=[]
with torch.no_grad():
    for m,j,n,y in valid_dl:
        logits=model(m.to(DEVICE),j.to(DEVICE),n.to(DEVICE))
        vt.extend(y.tolist()); vp.extend(logits.argmax(1).cpu().tolist())
acc=accuracy_score(vt,vp); mf1=f1_score(vt,vp,average="macro")
wf1=f1_score(vt,vp,average="weighted")
print("\n===== BEST VALIDATION CHECKPOINT =====")
print("Best epoch:",ckpt["best_epoch"])
print(f"Validation accuracy: {acc*100:.2f}%")
print(f"Macro-F1: {mf1:.4f} | Weighted-F1: {wf1:.4f}")
print(classification_report(vt,vp,labels=list(range(6)),target_names=LABELS,zero_division=0))
pd.DataFrame(classification_report(vt,vp,labels=list(range(6)),target_names=LABELS,output_dict=True,zero_division=0)).transpose().to_csv(REPORT)
print("Saved model:",best_path)
print("Saved report:",REPORT)
print("LIAR2 train/validation only; test split was not loaded.")

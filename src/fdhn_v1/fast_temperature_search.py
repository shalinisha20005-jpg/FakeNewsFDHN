import runpy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ns = runpy.run_path("src/fdhn_v1/evaluate_fdhn_ensemble.py")

original = ns["original_probs"]
frozen = ns["frozen_probs"]
y = ns["y"]

def scale_probs(probs, temperature):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    logits = logits / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

results = []

temperatures = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

for t1 in temperatures:
    p1 = scale_probs(original, t1)

    for t2 in temperatures:
        p2 = scale_probs(frozen, t2)

        for weight in np.arange(0.45, 0.751, 0.01):
            combined = weight * p1 + (1 - weight) * p2
            pred = combined.argmax(axis=1)

            results.append({
                "original_temperature": t1,
                "frozen_temperature": t2,
                "original_weight": round(weight, 2),
                "frozen_weight": round(1 - weight, 2),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro"),
                "weighted_f1": f1_score(y, pred, average="weighted")
            })

df = pd.DataFrame(results)
df = df.sort_values(
    ["accuracy", "macro_f1"],
    ascending=False
)

print("\n===== TOP 15 TEMPERATURE COMBINATIONS =====")
print(df.head(15).to_string(index=False))

best = df.iloc[0]

print("\n===== BEST RESULT =====")
print(f"Original temperature: {best.original_temperature}")
print(f"Frozen temperature: {best.frozen_temperature}")
print(f"Original weight: {best.original_weight}")
print(f"Frozen weight: {best.frozen_weight}")
print(f"Validation accuracy: {best.accuracy*100:.2f}%")
print(f"Macro-F1: {best.macro_f1:.4f}")
print(f"Weighted-F1: {best.weighted_f1:.4f}")

df.to_csv(
    "results/fdhn_temperature_search.csv",
    index=False
)

print("\nSaved results/fdhn_temperature_search.csv")
print("Validation set only. Test data was not used.")

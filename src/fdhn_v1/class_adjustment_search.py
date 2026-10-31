import runpy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ns = runpy.run_path("src/fdhn_v1/evaluate_fdhn_ensemble.py")

def scale_probs(probs, temperature):
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

p1 = scale_probs(ns["original_probs"], 1.0)
p2 = scale_probs(ns["frozen_probs"], 1.5)

base = 0.49 * p1 + 0.51 * p2
y = ns["y"]

# Try small boosts for the weaker classes.
# Class order: pants-fire, false, barely-true,
#              half-true, mostly-true, true
boosts = [1.00, 1.05, 1.10, 1.15, 1.20]

results = []

for pants in boosts:
    for barely in boosts:
        for true in boosts:
            multipliers = np.array([
                pants, 1.0, barely, 1.0, 1.0, true
            ])

            adjusted = base * multipliers
            adjusted /= adjusted.sum(axis=1, keepdims=True)

            pred = adjusted.argmax(axis=1)

            results.append({
                "pants_boost": pants,
                "barely_boost": barely,
                "true_boost": true,
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro")
            })

df = pd.DataFrame(results).sort_values(
    ["accuracy", "macro_f1"],
    ascending=False
)

print("\nTOP 15 CLASS ADJUSTMENTS")
print(df.head(15).to_string(index=False))

best = df.iloc[0]

print("\nBEST RESULT")
print(best.to_string())

df.to_csv(
    "results/fdhn_class_adjustment_search.csv",
    index=False
)

print("\nSaved results/fdhn_class_adjustment_search.csv")
print("Validation set only. Test data was not used.")

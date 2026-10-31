import runpy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

print("Loading existing ensemble predictions...")

# Execute the existing script to reconstruct the two models.
# This does not retrain them.
namespace = runpy.run_path("src/fdhn_v1/evaluate_fdhn_ensemble.py")

original_probs = namespace["original_probs"]
frozen_probs = namespace["frozen_probs"]
y = namespace["y"]

results = []

print("\nFine-grained ensemble weight search:")

for weight in np.arange(0.50, 0.801, 0.01):
    combined = (
        weight * original_probs
        + (1 - weight) * frozen_probs
    )

    pred = combined.argmax(axis=1)
    acc = accuracy_score(y, pred)
    macro = f1_score(y, pred, average="macro")
    weighted = f1_score(y, pred, average="weighted")

    results.append({
        "original_weight": round(weight, 2),
        "frozen_weight": round(1 - weight, 2),
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted
    })

results = pd.DataFrame(results)
results = results.sort_values(
    ["accuracy", "macro_f1"],
    ascending=False
)

print("\nTop 10 combinations:")
print(results.head(10).to_string(index=False))

best = results.iloc[0]

print("\n===== BEST FINE-GRAINED RESULT =====")
print(f"Original FDHN: {best['original_weight']:.2f}")
print(f"Frozen RoBERTa: {best['frozen_weight']:.2f}")
print(f"Validation accuracy: {best['accuracy']*100:.2f}%")
print(f"Macro-F1: {best['macro_f1']:.4f}")
print(f"Weighted-F1: {best['weighted_f1']:.4f}")

results.to_csv(
    "results/fdhn_fast_ensemble_search.csv",
    index=False
)

print("\nSaved: results/fdhn_fast_ensemble_search.csv")
print("Validation data only. Test data was not used.")

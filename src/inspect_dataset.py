
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/liar/source")

for filename in ["train.tsv", "valid.tsv", "test.tsv"]:
    file_path = DATA_DIR / filename

    print(f"\n--- {filename} ---")
    df = pd.read_csv(file_path, sep="\t", header=None)

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:", df.isnull().sum().sum())
    print("\nLabel counts:")
    print(df.iloc[:, 1].value_counts())

# Step 5: Inspect all training dataset columns

train_df = pd.read_csv(
    DATA_DIR / "train.tsv",
    sep="\t",
    header=None
)

print("\n--- COLUMN INSPECTION ---")

for column in train_df.columns:
    examples = train_df[column].dropna()

    if len(examples) > 0:
        example = examples.iloc[0]
    else:
        example = "All missing"

    print(
        f"Column {column}: "
        f"missing = {train_df[column].isnull().sum()}, "
        f"example = {example}"
    )


from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "liar" / "source"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

# Original LIAR column indexes.
# We deliberately exclude the speaker-history counts here.
TEXT_COLUMNS = {
    "statement": 2,
    "subject": 3,
    "speaker": 4,
    "job": 5,
    "state": 6,
    "party": 7,
    "context": 13,
}

for filename in ["train.tsv", "valid.tsv", "test.tsv"]:
    input_path = DATA_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path, sep="\t", header=None)

    if df.shape[1] < 14:
        raise ValueError(
            f"{filename} has {df.shape[1]} columns; "
            "the original LIAR dataset should have 14."
        )

    prepared = pd.DataFrame()
    prepared["label"] = df[1].astype("string").str.strip()

    # Build one model input from the claim and its context.
    for name, column_index in TEXT_COLUMNS.items():
        prepared[name] = (
            df[column_index]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    prepared = prepared.dropna(subset=["label"])
    prepared = prepared[prepared["label"].isin(LABELS)].copy()

    prepared["text"] = (
        "Statement: " + prepared["statement"]
        + " Subject: " + prepared["subject"]
        + " Speaker: " + prepared["speaker"]
        + " Job: " + prepared["job"]
        + " State: " + prepared["state"]
        + " Party: " + prepared["party"]
        + " Context: " + prepared["context"]
    )

    prepared["text"] = prepared["text"].str.strip()
    prepared = prepared[prepared["text"] != ""]

    output_path = OUTPUT_DIR / filename.replace(".tsv", ".csv")
    prepared[["text", "label"]].to_csv(output_path, index=False)

    print(f"\n{filename}")
    print("Prepared records:", len(prepared))
    print("Labels:", prepared["label"].value_counts().to_dict())
    print("Saved to:", output_path)

print("\nAll splits prepared using statement text and metadata.")

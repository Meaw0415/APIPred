"""Re-run the release audit; pandas and xlrd are required, no network is used."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aptamer_features import aptamer_features  # noqa: E402


def rows(frame):
    return Counter(map(tuple, frame.to_numpy()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dataset", type=Path, help="optional AptaNet Dataset.csv")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "train_Dataset.csv")
    test = pd.read_csv(ROOT / "test_Dataset.csv")
    schema = json.loads((ROOT / "docs/released_feature_schema.json").read_text())
    for name, frame in [("train", train), ("test", test)]:
        if frame.columns.tolist() != schema["pandas_columns"]:
            raise ValueError(name + " no longer has the audited release schema")
    with (ROOT / "train_Dataset.csv").open(newline="") as handle:
        raw_header = next(csv.reader(handle))
    if raw_header != schema["raw_csv_columns"]:
        raise ValueError("raw header no longer matches the audited release")
    combined = pd.concat([train, test], ignore_index=True)
    lookup = set(map(tuple, combined.iloc[:, 1:341].to_numpy()))
    raw_path = ROOT / "INPUT-APTAMER-PROTEIN-PAIRS.xls"
    raw = pd.read_excel(raw_path, header=1)
    eligible = 0
    matched = 0
    unique_matched = set()
    rejected = 0
    for sequence in raw["DNA/RNA sequence"]:
        try:
            feature = tuple(aptamer_features(sequence))
        except ValueError:
            rejected += 1
            continue
        eligible += 1
        if feature in lookup:
            matched += 1
            unique_matched.add(feature)
    report = {
        "baseline_commit": schema["source_commit"],
        "file_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["train_Dataset.csv", "test_Dataset.csv", raw_path.name]
        },
        "train_shape": list(train.shape), "test_shape": list(test.shape),
        "raw_sequence_rows": len(raw),
        "eligible_unambiguous_raw_rows": eligible,
        "rejected_raw_rows": rejected,
        "exact_aptamer_matches": matched,
        "distinct_matched_aptamer_vectors": len(unique_matched),
        "aptamer_match_scope": "any released aptamer vector; NOT a verified pair/label mapping",
        "full_640_sequence_reproduction_verified": False,
    }
    if args.upstream_dataset:
        upstream = pd.read_csv(args.upstream_dataset)
        if upstream.columns.tolist() != train.columns.tolist():
            raise ValueError("upstream schema differs from the release")
        upstream_rows = rows(upstream)
        report["upstream_comparison"] = {
            "sha256": hashlib.sha256(args.upstream_dataset.read_bytes()).hexdigest(),
            "shape": list(upstream.shape),
            "train_rows_absent_from_upstream": sum((rows(train) - upstream_rows).values()),
            "test_rows_absent_from_upstream": sum((rows(test) - upstream_rows).values()),
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

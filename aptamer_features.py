"""Numerically validated aptamer block; this is NOT a full APIPred extractor.

See docs/FEATURE_EXTRACTION.md for evidence, scope and unresolved protein details.
Python 3.9+; no third-party dependencies are required for this module.
"""

import argparse
from collections import Counter
import csv
from itertools import product
import json
from pathlib import Path


KMERS = tuple("".join(p) for k in range(1, 5) for p in product("ACGT", repeat=k))
SEMANTIC_COLUMNS = tuple("aptamer_frequency_" + word for word in KMERS)
SCHEMA_PATH = Path(__file__).resolve().parent / "docs/released_feature_schema.json"


def aptamer_features(sequence):
    """Return 340 positional frequencies, in lexical A,C,G,T order for k=1..4.

    Counts include overlapping windows. Each k uses denominator L-k+1;
    values are rounded to three decimal places. U is mapped to T.
    Lowercase and surrounding whitespace are accepted as an input convenience.
    N, B, gaps and other symbols are rejected: their historical treatment has
    not been recovered. Lengths below four are outside the validated scope.
    """
    if not isinstance(sequence, str):
        raise ValueError("aptamer sequence must be a string")
    sequence = sequence.strip().upper().replace("U", "T")
    invalid = set(sequence) - set("ACGT")
    if invalid:
        raise ValueError("unsupported aptamer symbols: " + ", ".join(sorted(invalid)))
    if len(sequence) < 4:
        raise ValueError("aptamer length must be at least 4")
    counts = {
        k: Counter(sequence[i:i + k] for i in range(len(sequence) - k + 1))
        for k in range(1, 5)
    }
    return [round(counts[len(w)][w] / (len(sequence) - len(w) + 1), 3) for w in KMERS]


def released_aptamer_columns():
    """Return the 340 names produced by pandas for the published CSV header.

    These legacy names are retained for positional compatibility. Use KMERS
    for the numerically supported k-mer at each position.
    """
    return json.loads(SCHEMA_PATH.read_text())["pandas_columns"][1:341]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with an aptamer_seq column")
    parser.add_argument("output", type=Path, help="340-column aptamer-only CSV")
    parser.add_argument("--sequence-column", default="aptamer_seq")
    parser.add_argument(
        "--released-header", action="store_true",
        help="retain the legacy pandas column names for positional compatibility",
    )
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different paths")
    # Validate the entire input before opening an output, so a bad sequence
    # cannot leave a partially written file that appears ready for prediction.
    rows = []
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if args.sequence_column not in (reader.fieldnames or []):
            parser.error("missing sequence column: " + args.sequence_column)
        for line, row in enumerate(reader, 2):
            try:
                rows.append(aptamer_features(row[args.sequence_column]))
            except ValueError as exc:
                parser.error(f"line {line}: {exc}")
    columns = released_aptamer_columns() if args.released_header else SEMANTIC_COLUMNS
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} aptamer blocks (340 columns). Protein features are unresolved;")
    print("this output is not a complete input for Xgboost.py.")


if __name__ == "__main__":
    main()

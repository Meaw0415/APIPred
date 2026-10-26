# APIPred

Code for [APIPred: An XGBoost-Based Method for Predicting Aptamer–Protein Interactions](https://doi.org/10.1021/acs.jcim.3c00713).

## Run the released models

From the repository root, with the model dependencies installed:

```sh
python Baseline.py
python Xgboost.py
```

`Xgboost.py` reads the processed `train_Dataset.csv` and `test_Dataset.csv`.
It trains feature selection, polynomial interactions and the classifier; it does
not extract features from sequences or load a pretrained inference pipeline.

## Custom data: current reproducibility status

**Changing CSV paths works only for already compatible processed features. A
verified end-to-end extractor from new sequence pairs to all 640 features is
not currently available.** The input has `Class` plus 340 aptamer features and
300 protein features, in the released positional order. The exact protein
preprocessing and fitted scaling parameters remain unresolved.

We provide a numerically checked **aptamer-only** extractor for unambiguous
sequences, including RNA with U mapped to T:

```csv
aptamer_seq
ACGTACGT
ACGUACGU
```

Save this as `pairs.csv`, then run with Python 3.9+ (no extra dependencies):

```sh
python aptamer_features.py pairs.csv aptamer_only.csv
# Optional: match the names pandas assigns to the published aptamer columns.
python aptamer_features.py pairs.csv aptamer_only.csv --released-header
```

The output contains 340 aptamer columns and cannot by itself be passed to
`Xgboost.py`. N/B, gaps and unsupported symbols are rejected because their
historical processing is unverified. Use the original positional order for
compatibility; the [mapping](docs/aptamer_column_mapping.tsv) records the legacy
column names and the corresponding numerically supported k-mers.

See [feature extraction and data provenance](docs/FEATURE_EXTRACTION.md) for
the validation results, limitations, source papers and original code links.

## Tests and reproducibility audit

The basic tests use only the Python standard library:

```sh
python -m unittest discover -s tests -v
```

The broader audit also reads the released XLS and requires pandas and xlrd:

```sh
python -m pip install pandas xlrd
python scripts/audit_features.py
```

An optional upstream comparison and the pinned source version are described
in the [audit documentation](docs/FEATURE_EXTRACTION.md#repeat-the-audit).

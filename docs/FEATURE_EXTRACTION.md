# Feature extraction and data sources

This note supports [Issue #1](https://github.com/Meaw0415/APIPred/issues/1).
It describes the added aptamer helper and the input format used by
`Xgboost.py`. Validation uses the released files at commit
`d751e5732c3fa957d2ae416e64119651037def35`.

## Input format

The released CSV files contain `Class`, followed by 340 aptamer values and
300 protein values. A labeled input therefore has 641 columns. `Xgboost.py`
reads these processed values; it does not extract features from sequences.

| Positions excluding Class | Contents |
| --- | --- |
| 1–4 | Aptamer 1-mers |
| 5–20 | Aptamer 2-mers |
| 21–84 | Aptamer 3-mers |
| 85–340 | Aptamer 4-mers |
| 341–640 | Six protein blocks, each with 20 Xc1 and 30 Xc2.lambda columns |

Keep the original positional order. The
[schema](released_feature_schema.json) records both the literal CSV header and
pandas column names. Suffixes `.1`–`.5` are assigned by pandas to repeated
protein headers; they do not by themselves identify the property groups.
The [aptamer mapping](aptamer_column_mapping.tsv) lists the legacy names and
numerically supported k-mer for each position.

## Aptamer helper

For an unambiguous sequence of length L, `aptamer_features.py`:

1. Maps U to T.
2. Counts overlapping windows for each k = 1, 2, 3, 4.
3. Divides each count by L − k + 1 and rounds to three decimal places.
4. Concatenates the blocks in lexical A, C, G, T order.

This gives 4 + 16 + 64 + 256 = 340 values. The helper uses ordinary k-mer
counts, without a separate or merged reverse-complement block. Values are not
renormalized after rounding.

Across the raw XLS, all 340 values match a released aptamer vector exactly
for 1,958 records, covering 479 distinct vectors. This comparison checks the
aptamer block against any released row; it does not establish a complete
sequence-pair mapping. The report includes the validation scope and file
digests in [feature_audit.json](feature_audit.json).

Save a CSV with an `aptamer_seq` column, then run:

```sh
python aptamer_features.py pairs.csv aptamer_only.csv
python aptamer_features.py pairs.csv aptamer_only.csv --released-header
```

The default output uses semantic k-mer names. `--released-header` retains the
legacy pandas names at the same positions. **Both outputs are aptamer-only;
they do not provide the 300 protein features needed by the model.**

Lowercase and surrounding whitespace are accepted as input conveniences.
N/B, gaps, other unsupported symbols, and lengths below four are rejected.
The paper discusses multiple N/B handling strategies; their original
per-record settings have not been recovered, so the helper does not guess.

## Protein features

The complete historical sequence-to-feature script and fitted preprocessing
parameters have not been located in the public repository or its history.
The lambda indices 1–30 are visible in the release, but the original package,
weight, property tables, group assignment and scaling are not yet sufficiently
verified to provide a compatible 300-feature extractor.

For reference, AptaNet provides
[feature_extraction.py](https://github.com/nedaemami/AptaNet/blob/8f1f4d1fd13c97b84b828e517f7e1355366a7f07/feature_extraction.py)
and
[target_encoding.py](https://github.com/nedaemami/AptaNet/blob/8f1f4d1fd13c97b84b828e517f7e1355366a7f07/target_encoding.py).
These examples use lambda = 30 and weight = 0.15, but have not been verified as
reproducing this release. APIPred also retains an earlier
[data-processing utility](https://github.com/Meaw0415/APIPred/blob/53d1df3/mlpcnnsemi/Network/API_utils/data_process.py)
in its history; it is not the complete 340 + 300 pipeline.

Until the remaining preprocessing is recovered, the helper should not be
presented as a full raw-sequence predictor. The existing model script accepts
custom CSV paths when the features are already compatible.

## Data sources

- **Processed tables:** every released training/test row matches all 641
  numeric values of a row in
  [AptaNet's Dataset.csv](https://github.com/nedaemami/AptaNet/blob/8f1f4d1fd13c97b84b828e517f7e1355366a7f07/Dataset.csv).
  The same complete file is preserved in
  [APIPred's history](https://github.com/Meaw0415/APIPred/blob/53d1df3/mlpcnnsemi/dataset/Dataset.csv),
  with Git blob SHA `7da1c5c255848aaa5bc521444df592c9bc784d14`.
  See Emami and Ferdousi,
  [AptaNet](https://doi.org/10.1038/s41598-021-85629-0), Scientific Reports (2021).
- **Raw sequence pairs:** the first cell of
  `INPUT-APTAMER-PROTEIN-PAIRS.xls` states that it is adapted from Li et al.,
  [PPAI](https://doi.org/10.1186/s12859-020-03574-7), BMC Bioinformatics (2020).
  Its Supplementary Table S5 is a source of raw sequence-pair data.

These identify the available source resources; the original complete
raw-to-processed row mapping has not been recovered.

## Repeat the audit

The basic tests use the standard library. The XLS audit requires pandas and
xlrd and makes no network requests:

```sh
python -m unittest discover -s tests -v
python -m pip install pandas xlrd
python scripts/audit_features.py
```

To include the optional upstream comparison:

```sh
git clone https://github.com/nedaemami/AptaNet.git ../aptanet-source
git -C ../aptanet-source checkout 8f1f4d1fd13c97b84b828e517f7e1355366a7f07
python scripts/audit_features.py --upstream-dataset ../aptanet-source/Dataset.csv
```

The tests include full 340-value comparisons to three training and three test
records, overlapping-window counts, RNA conversion, legacy headers, validation
errors and CLI output. No model retraining is required.

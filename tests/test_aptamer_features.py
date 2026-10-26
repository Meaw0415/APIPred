import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from aptamer_features import (
    aptamer_features, KMERS, main, released_aptamer_columns, SEMANTIC_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[1]


class AptamerFeatureTests(unittest.TestCase):
    def test_overlapping_windows_and_per_k_denominators(self):
        values = dict(zip(KMERS, aptamer_features("ATATAT")))
        self.assertEqual(values["ATA"], 0.5)
        self.assertEqual(values["TAT"], 0.5)
        self.assertEqual(values["ATAT"], 0.667)
        self.assertEqual(values["TATA"], 0.333)
        self.assertEqual(dict(zip(KMERS, aptamer_features("AAAAA")))["AAAA"], 1.0)

    def test_rna_and_input_conveniences(self):
        self.assertEqual(aptamer_features(" auguau \n"), aptamer_features("ATGTAT"))

    def test_unknown_historical_rules_fail_explicitly(self):
        for sequence in ["ACGN", "ACGB", "AC-G", "ACG", "", None]:
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                aptamer_features(sequence)

    def test_no_reverse_complement_merging(self):
        values = dict(zip(KMERS, aptamer_features("AAAAC")))
        self.assertEqual(values["AAAA"], 0.5)
        self.assertEqual(values["TTTT"], 0.0)

    def test_exact_full_block_regression_against_both_release_tables(self):
        fixtures = json.loads((ROOT / "tests/fixtures/aptamer_regression.json").read_text())
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with (ROOT / fixture["feature_table"]).open(newline="") as handle:
                    reader = csv.reader(handle)
                    next(reader)
                    row = next(r for i, r in enumerate(reader) if i == fixture["feature_row_0based"])
                self.assertEqual(aptamer_features(fixture["aptamer_seq"]), list(map(float, row[1:341])))

    def test_released_names_are_positional_compatibility_labels(self):
        columns = released_aptamer_columns()
        self.assertEqual(len(columns), 340)
        self.assertEqual(columns[75], "aptamer_frequency_TTT")
        self.assertEqual(SEMANTIC_COLUMNS[75], "aptamer_frequency_TCT")
        self.assertEqual(columns[83], "aptamer_frequency_TTT.1")

    def test_cli_emits_only_aptamer_block_and_preserves_existing_output_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "input.csv", Path(tmp) / "features.csv"
            source.write_text("aptamer_seq\nACGTAC\n")
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(source), str(dest), "--released-header"])
            with dest.open(newline="") as handle:
                header, values = list(csv.reader(handle))
            self.assertEqual(header, released_aptamer_columns())
            self.assertEqual(len(values), 340)
            old = dest.read_bytes()
            source.write_text("aptamer_seq\nACGTAC\nACGN\n")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(source), str(dest)])
            self.assertEqual(dest.read_bytes(), old)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(source), str(source)])


if __name__ == "__main__":
    unittest.main()

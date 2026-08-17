import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from raw_data_processing.compare_coarsening_audits import (
    OUTPUT_MASS_LIMITATION,
    PSEUDO_PARTITION_LIMITATION,
    SFR_LIMITATION,
    build_parser,
    compare_audits,
)


class CompareCoarseningAuditsTests(unittest.TestCase):
    def _write_inputs(self, directory: Path):
        common_metrics = {
            "epsilon_conservation_max": 1e-12,
            "epsilon_balance": 1e-7,
            "balance_iterations": 100,
            "source_enclosed_mass_to_filter_median": 1.05,
            "source_enclosed_mass_to_filter_p95": 1.3,
            "source_enclosed_mass_to_filter_max": 2.0,
            "output_cell_mass_cv": 0.8,
            "output_cell_mass_gini": 0.4,
            "h_eff_median": 2.0,
            "h_eff_p95": 3.0,
            "n_eff_median": 10.0,
            "n_eff_p95": 20.0,
            "dominant_partition_component_total": 7,
            "dominant_partition_disconnected_count": 2,
            "dominant_partition_empty_count": 0,
            "dominant_partition_largest_component_fraction_mean": 0.85,
            "resolved_kinetic_energy_loss_fraction": 0.15,
            "stored_extensive_conservation_pass": True,
        }
        v0 = pd.DataFrame(
            [
                {"sample_uid": "A", "status": "ok", "runtime_wall_seconds": 10.0, **common_metrics},
                {"sample_uid": "B", "status": "timeout", "error_type": "TimeoutError"},
                {"sample_uid": "C", "status": "failed", "error_type": "RuntimeError"},
                {"sample_uid": "D", "status": "ok", "runtime_wall_seconds": 20.0, **common_metrics},
                {"sample_uid": "V0_ONLY", "status": "ok", "runtime_wall_seconds": 5.0, **common_metrics},
            ]
        )
        v1_metrics = {
            **common_metrics,
            "epsilon_balance": 5e-8,
            "balance_iterations": 40,
            "output_cell_mass_cv": 0.7,
            "dominant_partition_disconnected_count": 1,
            "resolved_kinetic_energy_loss_fraction": 0.10,
        }
        v1 = pd.DataFrame(
            [
                {"sample_uid": "A", "status": "ok", "runtime_wall_seconds": 4.0, **v1_metrics},
                {"sample_uid": "B", "status": "ok", "runtime_wall_seconds": 3.0, **v1_metrics},
                {"sample_uid": "C", "status": "timeout", "error_type": "TimeoutError"},
                {"sample_uid": "D", "status": "ok", "runtime_wall_seconds": 10.0, **v1_metrics},
                {"sample_uid": "V1_ONLY", "status": "ok", "runtime_wall_seconds": 2.0, **v1_metrics},
            ]
        )
        v0_path = directory / "v0.csv"
        v1_path = directory / "v1.csv"
        v0.to_csv(v0_path, index=False)
        v1.to_csv(v1_path, index=False)
        return v0_path, v1_path

    def test_paired_outputs_keep_unsuccessful_common_rows_visible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            v0_path, v1_path = self._write_inputs(directory)
            output = directory / "comparison"
            summary = compare_audits(v0_path, v1_path, output)

            paired = pd.read_csv(output / "comparison_per_snapshot.csv")
            self.assertEqual(set(paired["sample_uid"]), {"A", "B", "C", "D"})
            self.assertEqual(int(paired["both_success"].sum()), 2)
            row_b = paired.set_index("sample_uid").loc["B"]
            self.assertFalse(bool(row_b["both_success"]))
            self.assertTrue(pd.isna(row_b["delta_v1_minus_v0_epsilon_balance"]))
            row_a = paired.set_index("sample_uid").loc["A"]
            self.assertAlmostEqual(row_a["runtime_speedup_v0_over_v1"], 2.5)
            self.assertAlmostEqual(
                row_a["delta_v1_minus_v0_output_cell_mass_cv"], -0.1
            )

            coverage = summary["coverage"]
            self.assertEqual(coverage["common_sample_uid_count"], 4)
            self.assertEqual(coverage["both_success_count"], 2)
            self.assertEqual(coverage["common_but_not_both_success_count"], 2)
            self.assertEqual(coverage["v0_only_sample_uids"], ["V0_ONLY"])
            self.assertEqual(coverage["v1_only_sample_uids"], ["V1_ONLY"])
            self.assertEqual(coverage["v0"]["timeout"], 1)
            self.assertEqual(coverage["v0"]["failed"], 1)
            self.assertEqual(coverage["v1"]["timeout"], 1)

            with (output / "comparison_summary.json").open("r", encoding="utf-8") as stream:
                restored = json.load(stream)
            self.assertIn(SFR_LIMITATION, restored["limitations"])
            self.assertIn(PSEUDO_PARTITION_LIMITATION, restored["limitations"])
            self.assertIn(OUTPUT_MASS_LIMITATION, restored["limitations"])
            report = (output / "comparison_report.md").read_text(encoding="utf-8")
            self.assertIn("不能给出SFR优劣结论", report)
            self.assertIn("不是等质量目标", report)

    def test_duplicate_sample_uid_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            v0_path, v1_path = self._write_inputs(directory)
            duplicate = pd.read_csv(v0_path)
            duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
            duplicate.to_csv(v0_path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate sample_uid"):
                compare_audits(v0_path, v1_path, directory / "out")

    def test_optional_output_cells_fill_missing_summary_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            v0_summary = directory / "v0_summary.csv"
            v1_summary = directory / "v1_summary.csv"
            pd.DataFrame(
                [{"sample_uid": "A", "status": "ok", "runtime_wall_seconds": 2.0}]
            ).to_csv(v0_summary, index=False)
            pd.DataFrame(
                [{"sample_uid": "A", "status": "ok", "runtime_wall_seconds": 1.0}]
            ).to_csv(v1_summary, index=False)
            v0_cells = directory / "v0_cells.csv"
            v1_cells = directory / "v1_cells.csv"
            pd.DataFrame(
                [
                    {
                        "sample_uid": "A",
                        "output_cell_mass": 1.0,
                        "effective_support_scale": 2.0,
                        "effective_source_count": 4.0,
                        "dominant_component_count": 1,
                        "dominant_assigned_source_count": 2,
                        "dominant_largest_component_fraction": 1.0,
                    },
                    {
                        "sample_uid": "A",
                        "output_cell_mass": 3.0,
                        "effective_support_scale": 4.0,
                        "effective_source_count": 8.0,
                        "dominant_component_count": 2,
                        "dominant_assigned_source_count": 2,
                        "dominant_largest_component_fraction": 0.5,
                    },
                ]
            ).to_csv(v0_cells, index=False)
            pd.DataFrame(
                [
                    {
                        "sample_uid": "A",
                        "output_cell_mass": 2.0,
                        "effective_support_scale": 1.0,
                        "effective_source_count": 5.0,
                        "dominant_component_count": 1,
                        "dominant_assigned_source_count": 2,
                        "dominant_largest_component_fraction": 1.0,
                    },
                    {
                        "sample_uid": "A",
                        "output_cell_mass": 2.0,
                        "effective_support_scale": 3.0,
                        "effective_source_count": 7.0,
                        "dominant_component_count": 1,
                        "dominant_assigned_source_count": 2,
                        "dominant_largest_component_fraction": 1.0,
                    },
                ]
            ).to_csv(v1_cells, index=False)
            output = directory / "out"
            compare_audits(
                v0_summary,
                v1_summary,
                output,
                v0_cells_path=v0_cells,
                v1_cells_path=v1_cells,
            )
            row = pd.read_csv(output / "comparison_per_snapshot.csv").iloc[0]
            self.assertAlmostEqual(row["v0_output_cell_mass_cv"], 0.5)
            self.assertAlmostEqual(row["v1_output_cell_mass_cv"], 0.0)
            self.assertAlmostEqual(row["v0_h_eff_median"], 3.0)
            self.assertAlmostEqual(row["v1_h_eff_median"], 2.0)
            self.assertEqual(row["v0_dominant_partition_disconnected_count"], 1)
            self.assertEqual(row["v1_dominant_partition_disconnected_count"], 0)

    def test_cli_help(self):
        help_text = build_parser().format_help()
        self.assertIn("--v0-summary", help_text)
        self.assertIn("--v1-summary", help_text)
        self.assertIn("does not compare SFR", help_text)


if __name__ == "__main__":
    unittest.main()

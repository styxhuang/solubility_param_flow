"""Tests for ML feature preparation."""

from pathlib import Path

import pandas as pd

from solubility_param_flow.models.feature_builder import HSPFeatureBuilder


def test_feature_builder_filters_success_rows_and_builds_combined_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "workflow_results.csv"
    pd.DataFrame(
        [
            {
                "status": "success",
                "mw": 46.07,
                "logp": -0.1,
                "tpsa": 20.23,
                "hbd": 1,
                "hba": 1,
                "rotatable_bonds": 0,
                "fraction_csp3": 1.0,
                "heavy_atom_count": 3,
                "ring_count": 0,
                "hetero_atom_count": 1,
                "sigma_moment_2": 0.4,
                "sigma_moment_3": 0.1,
                "sigma_moment_4": 0.2,
                "hb_acceptor": 1.0,
                "hb_donor": 1.0,
                "cavity_volume": 40.0,
                "delta_d": 15.5,
                "delta_p": 8.2,
                "delta_h": 18.0,
            },
            {
                "status": "invalid_smiles",
                "delta_d": 0.0,
                "delta_p": 0.0,
                "delta_h": 0.0,
            },
        ]
    ).to_csv(csv_path, index=False)

    builder = HSPFeatureBuilder()
    success_rows = builder.load_success_rows(str(csv_path))
    x_frame, y_frame = builder.build_feature_frame(success_rows, "combined")

    assert len(success_rows) == 1
    assert list(y_frame.columns) == ["delta_d", "delta_p", "delta_h"]
    assert "mw" in x_frame.columns
    assert "sigma_moment_2" in x_frame.columns

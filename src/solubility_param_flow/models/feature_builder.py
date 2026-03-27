"""Feature builders for HSP modeling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HSPFeatureBuilder:
    """Build feature matrices from pipeline result CSV files."""

    rdkit_columns = [
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "fraction_csp3",
        "heavy_atom_count",
        "ring_count",
        "hetero_atom_count",
    ]
    cosmo_columns = [
        "sigma_moment_2",
        "sigma_moment_3",
        "sigma_moment_4",
        "hb_acceptor",
        "hb_donor",
        "cavity_volume",
    ]
    target_columns = ["delta_d", "delta_p", "delta_h"]

    def load_success_rows(self, csv_path: str) -> pd.DataFrame:
        frame = pd.read_csv(csv_path)
        success_frame = frame[frame["status"] == "success"].copy()
        if success_frame.empty:
            raise ValueError("No successful rows found in the workflow result file.")
        return success_frame

    def build_feature_frame(self, frame: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_columns = self._resolve_feature_columns(feature_set)
        required_columns = feature_columns + self.target_columns
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns for feature set `{feature_set}`: {missing}")

        usable_frame = frame.dropna(subset=required_columns).copy()
        if usable_frame.empty:
            raise ValueError(f"No rows remain after dropping missing values for `{feature_set}`")

        return usable_frame[feature_columns], usable_frame[self.target_columns]

    def _resolve_feature_columns(self, feature_set: str) -> list[str]:
        normalized = feature_set.lower()
        if normalized == "rdkit":
            return self.rdkit_columns
        if normalized == "cosmo":
            return self.cosmo_columns
        if normalized == "combined":
            return self.rdkit_columns + self.cosmo_columns
        raise ValueError(f"Unsupported feature set: {feature_set}")

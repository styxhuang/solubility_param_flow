"""CSV loading and SMILES preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from solubility_param_flow.schemas import MoleculeRecord


@dataclass
class CsvSmilesLoader:
    """Load a CSV file and normalize its SMILES records."""

    smiles_column: str = "smiles"
    name_column: Optional[str] = "name"

    def load(self, csv_path: str) -> Tuple[List[MoleculeRecord], pd.DataFrame]:
        """Return valid molecule records and invalid rows."""
        frame = pd.read_csv(csv_path)
        if self.smiles_column not in frame.columns:
            raise ValueError(
                f"CSV must contain a `{self.smiles_column}` column, found: {list(frame.columns)}"
            )

        records: List[MoleculeRecord] = []
        invalid_rows: list[dict[str, object]] = []

        for row_index, row in frame.iterrows():
            input_smiles = str(row[self.smiles_column]).strip()
            if not input_smiles or input_smiles.lower() == "nan":
                invalid_rows.append(
                    {
                        "row_index": row_index,
                        "input_smiles": input_smiles,
                        "error_message": "Empty SMILES value",
                    }
                )
                continue

            canonical_smiles = self._canonicalize_smiles(input_smiles)
            if canonical_smiles is None:
                invalid_rows.append(
                    {
                        "row_index": row_index,
                        "input_smiles": input_smiles,
                        "error_message": "Invalid SMILES",
                    }
                )
                continue

            name = self._resolve_name(row, row_index, canonical_smiles)
            metadata = {
                key: value
                for key, value in row.to_dict().items()
                if key not in {self.smiles_column, self.name_column}
            }
            records.append(
                MoleculeRecord(
                    molecule_id=f"mol_{row_index:05d}",
                    row_index=row_index,
                    input_smiles=input_smiles,
                    canonical_smiles=canonical_smiles,
                    name=name,
                    metadata=metadata,
                )
            )

        return records, pd.DataFrame(invalid_rows)

    @staticmethod
    def _canonicalize_smiles(smiles: str) -> Optional[str]:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def _resolve_name(self, row: pd.Series, row_index: int, canonical_smiles: str) -> str:
        if self.name_column and self.name_column in row and pd.notna(row[self.name_column]):
            candidate = str(row[self.name_column]).strip()
            if candidate:
                return candidate
        return f"molecule_{row_index}_{canonical_smiles}"

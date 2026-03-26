"""End-to-end dry-run pipeline from CSV/SMILES to mock HSP outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from solubility_param_flow.data import CsvSmilesLoader
from solubility_param_flow.quantum import DryRunOpenCosmoRunner, DryRunOrcaRunner
from solubility_param_flow.schemas import PipelineRecord


class SmilesToHSPDryRunPipeline:
    """Build workflow artifacts without calling local ORCA or OpenCOSMO-RS."""

    def __init__(
        self,
        smiles_column: str = "smiles",
        name_column: str = "name",
        orca_binary: str = "/root/orca600/orca",
    ):
        self.loader = CsvSmilesLoader(smiles_column=smiles_column, name_column=name_column)
        self.orca_runner = DryRunOrcaRunner(orca_binary=orca_binary)
        self.opencosmo_runner = DryRunOpenCosmoRunner()

    def run(self, csv_path: str, output_dir: str) -> pd.DataFrame:
        base_dir = Path(output_dir)
        preprocessed_dir = base_dir / "preprocessed"
        jobs_dir = base_dir / "jobs"
        results_dir = base_dir / "results"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        valid_records, invalid_df = self.loader.load(csv_path)
        valid_frame = pd.DataFrame([record.model_dump() for record in valid_records])
        valid_frame.to_csv(preprocessed_dir / "valid_molecules.csv", index=False)
        invalid_df.to_csv(preprocessed_dir / "invalid_molecules.csv", index=False)

        outputs: List[PipelineRecord] = []
        for record in valid_records:
            try:
                orca_result = self.orca_runner.prepare(record, str(jobs_dir))
                cosmo_result = self.opencosmo_runner.prepare(record, orca_result)
                outputs.append(
                    PipelineRecord(
                        molecule=record,
                        status="success",
                        orca=orca_result,
                        cosmo=cosmo_result,
                    )
                )
            except Exception as exc:
                outputs.append(
                    PipelineRecord(
                        molecule=record,
                        status="failed",
                        error_message=str(exc),
                    )
                )

        for _, row in invalid_df.iterrows():
            outputs.append(
                PipelineRecord(
                    molecule={
                        "molecule_id": f"mol_{int(row['row_index']):05d}",
                        "row_index": int(row["row_index"]),
                        "input_smiles": str(row["input_smiles"]),
                        "canonical_smiles": "",
                        "name": f"invalid_{int(row['row_index'])}",
                        "metadata": {},
                    },
                    status="invalid_smiles",
                    error_message=str(row["error_message"]),
                )
            )

        result_frame = pd.DataFrame([self._flatten_record(record) for record in outputs])
        result_frame = result_frame.sort_values(["row_index", "molecule_id"]).reset_index(drop=True)
        result_frame.to_csv(results_dir / "hsp_workflow_results.csv", index=False)
        return result_frame

    @staticmethod
    def _flatten_record(record: PipelineRecord) -> dict[str, object]:
        flattened = {
            "molecule_id": record.molecule.molecule_id,
            "row_index": record.molecule.row_index,
            "name": record.molecule.name,
            "input_smiles": record.molecule.input_smiles,
            "canonical_smiles": record.molecule.canonical_smiles,
            "status": record.status,
            "error_message": record.error_message,
        }

        if record.orca is not None:
            flattened.update(
                {
                    "orca_mode": record.orca.mode,
                    "orca_workdir": record.orca.workdir,
                    "orca_command": record.orca.command,
                    "opt_input_path": record.orca.opt_input_path,
                    "sp_input_path": record.orca.sp_input_path,
                    "cosmo_input_path": record.orca.cosmo_input_path,
                }
            )

        if record.cosmo is not None:
            flattened.update(
                {
                    "hsp_source": record.cosmo.hsp_source,
                    "delta_d": record.cosmo.delta_d,
                    "delta_p": record.cosmo.delta_p,
                    "delta_h": record.cosmo.delta_h,
                    "sigma_moment_2": record.cosmo.descriptors.get("sigma_moment_2"),
                    "sigma_moment_3": record.cosmo.descriptors.get("sigma_moment_3"),
                    "sigma_moment_4": record.cosmo.descriptors.get("sigma_moment_4"),
                    "hb_acceptor": record.cosmo.descriptors.get("hb_acceptor"),
                    "hb_donor": record.cosmo.descriptors.get("hb_donor"),
                    "cavity_volume": record.cosmo.descriptors.get("cavity_volume"),
                    "opencosmo_result_path": record.cosmo.result_path,
                }
            )

        return flattened

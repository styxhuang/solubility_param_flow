"""Tests for CSV preprocessing and dry-run pipeline."""

import json
from pathlib import Path

import pandas as pd

from solubility_param_flow.descriptors.molecular_descriptor import DescriptorCalculator
from solubility_param_flow.pipelines import SmilesToHSPDryRunPipeline
from solubility_param_flow.schemas import WorkflowExecutionSettings


def test_descriptor_calculator_uses_rdkit() -> None:
    calculator = DescriptorCalculator()
    descriptor = calculator.calculate("CCO")

    assert descriptor.mw > 40
    assert descriptor.tpsa > 0
    assert descriptor.descriptors["heavy_atom_count"] == 3.0


def test_smiles_pipeline_writes_dry_run_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame(
        [
            {"name": "ethanol", "smiles": "CCO"},
            {"name": "bad", "smiles": "not_a_smiles"},
        ]
    ).to_csv(csv_path, index=False)

    output_dir = tmp_path / "artifacts"
    settings = WorkflowExecutionSettings()
    pipeline = SmilesToHSPDryRunPipeline(settings=settings)
    result_frame = pipeline.run(str(csv_path), str(output_dir))

    assert len(result_frame) == 2

    success_row = result_frame[result_frame["status"] == "success"].iloc[0]
    assert success_row["canonical_smiles"] == "CCO"
    assert "/root/orca600/orca" in success_row["orca_command"]
    assert "OMPI_ALLOW_RUN_AS_ROOT=1" in success_row["orca_command"]
    assert "bohr job submit" in success_row["remote_submit_command"]
    assert success_row["delta_d"] > 0
    assert success_row["sigma_moment_2"] > 0

    invalid_row = result_frame[result_frame["status"] == "invalid_smiles"].iloc[0]
    assert invalid_row["error_message"] == "Invalid SMILES"

    manifest_path = output_dir / "results" / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["orca"]["mode"] == "dry-run"
    assert manifest["opencosmo"]["reference_url"].endswith("openCOSMO-RS_py")

    assert (output_dir / "preprocessed" / "valid_molecules.csv").exists()
    assert (output_dir / "preprocessed" / "invalid_molecules.csv").exists()
    assert (output_dir / "results" / "hsp_workflow_results.csv").exists()
    assert manifest_path.exists()

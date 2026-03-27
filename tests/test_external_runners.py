"""Tests for external Uni-Mol and uni-elf runners."""

import json
from pathlib import Path

from solubility_param_flow.external import ExternalModelSettings, UniElfRunner, UniMolRunner


def test_unielf_runner_writes_training_manifest(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("smiles,delta_d\nCCO,15.5\n", encoding="utf-8")

    settings = ExternalModelSettings()
    runner = UniElfRunner(settings=settings)
    artifacts = runner.prepare_training_job(str(dataset_csv), str(tmp_path))

    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["backend"] == "uni-elf"
    assert "bohr job submit" in manifest["command"]
    assert "python run_unielf_train.py" in manifest["remote_command"]
    assert manifest["machine_type"] == "c4_m8_cpu"
    assert manifest["use_job_group"] is False
    assert Path(artifacts["config_path"]).exists()
    assert Path(artifacts["script_path"]).exists()
    assert (tmp_path / "unielf" / "dataset.csv").exists()


def test_unielf_runner_supports_existing_job_group(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("smiles,delta_d\nCCO,15.5\n", encoding="utf-8")

    settings = ExternalModelSettings()
    settings.unielf.bohrium.use_job_group = True
    settings.unielf.bohrium.job_group_id = 12345
    runner = UniElfRunner(settings=settings)
    artifacts = runner.prepare_training_job(str(dataset_csv), str(tmp_path))

    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["job_group_id"] == 12345
    assert manifest["job_group_name"] is None
    assert "-g 12345" in manifest["command"]


def test_unimol_runner_writes_bootstrap_script_and_job_group(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("smiles,delta_d,delta_p,delta_h\nCCO,15.5,8.2,18.0\n", encoding="utf-8")

    settings = ExternalModelSettings()
    runner = UniMolRunner(settings=settings)
    artifacts = runner.prepare_training_job(str(dataset_csv), str(tmp_path))

    script_text = Path(artifacts["script_path"]).read_text(encoding="utf-8")
    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    assert "UniMolRepr" in script_text
    assert "bohr job submit" in artifacts["command"]
    assert "bohr job_group create" in artifacts["command"]
    assert manifest["machine_type"] == "c16_m62_1 * NVIDIA T4"
    assert manifest["use_job_group"] is True
    assert manifest["job_group_name"] == "unimol-train"
    assert "python run_unimol.py" in manifest["remote_command"]
    assert Path(artifacts["script_path"]).exists()
    assert (tmp_path / "unimol" / "dataset.csv").exists()

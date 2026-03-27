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
    assert "unielf-train" in manifest["command"]
    assert "conda activate" in manifest["command"]
    assert Path(artifacts["config_path"]).exists()


def test_unimol_runner_writes_bootstrap_script(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("smiles,delta_d,delta_p,delta_h\nCCO,15.5,8.2,18.0\n", encoding="utf-8")

    settings = ExternalModelSettings()
    runner = UniMolRunner(settings=settings)
    artifacts = runner.prepare_training_job(str(dataset_csv), str(tmp_path))

    script_text = Path(artifacts["script_path"]).read_text(encoding="utf-8")
    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    assert "unimol_tools" in script_text or "unimol" in manifest["backend"]
    assert "HTTP_PROXY" in artifacts["command"]
    assert Path(artifacts["script_path"]).exists()

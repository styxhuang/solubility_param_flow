"""Command builders for external Uni-Mol and uni-elf workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from solubility_param_flow.external.config import ExternalModelSettings


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_dump(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    for key, value in data.items():
        prefix = " " * indent + f"{key}:"
        if isinstance(value, dict):
            lines.append(prefix)
            lines.append(_yaml_dump(value, indent=indent + 2))
        elif isinstance(value, list):
            lines.append(prefix)
            for item in value:
                if isinstance(item, dict):
                    lines.append(" " * (indent + 2) + "-")
                    lines.append(_yaml_dump(item, indent=indent + 4))
                else:
                    lines.append(" " * (indent + 2) + f"- {_yaml_scalar(item)}")
        else:
            lines.append(f"{prefix} {_yaml_scalar(value)}")
    return "\n".join(lines)


class UniElfRunner:
    """Prepare config files and shell commands for uni-elf."""

    def __init__(self, settings: ExternalModelSettings | None = None):
        self.settings = settings or ExternalModelSettings()

    def prepare_training_job(
        self,
        dataset_csv: str,
        output_dir: str,
        task: str = "downstream_single",
        target_column: str = "delta_d",
        metrics: str = "mae",
    ) -> dict[str, str]:
        target_dir = Path(output_dir) / "unielf"
        target_dir.mkdir(parents=True, exist_ok=True)

        config_path = target_dir / "train_config.yaml"
        config_payload = {
            "task": task,
            "data_path": dataset_csv,
            "target_col": target_column,
            "metrics": metrics,
            "batch_size": 16,
            "max_epoch": 20,
            "tensorboard_logdir": str(target_dir / "tsb"),
            "save_dir": str(target_dir / "ckpts"),
            "tmp_save_dir": str(target_dir / "ckpts"),
        }
        config_path.write_text(_yaml_dump(config_payload) + "\n", encoding="utf-8")

        manifest_path = target_dir / "train_manifest.json"
        manifest = {
            "backend": "uni-elf",
            "dataset_csv": dataset_csv,
            "config_path": str(config_path),
            "command": self.build_train_command(str(config_path)),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "config_path": str(config_path),
            "manifest_path": str(manifest_path),
            "command": manifest["command"],
        }

    def prepare_inference_job(
        self,
        dataset_csv: str,
        model_file: str,
        output_dir: str,
        config_path: str,
        scaler_path: str | None = None,
    ) -> dict[str, str]:
        target_dir = Path(output_dir) / "unielf"
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "inference_manifest.json"
        command = self.build_inference_command(
            model_file=model_file,
            dataset_csv=dataset_csv,
            config_path=config_path,
            scaler_path=scaler_path,
        )
        manifest = {
            "backend": "uni-elf",
            "dataset_csv": dataset_csv,
            "model_file": model_file,
            "config_path": config_path,
            "scaler_path": scaler_path,
            "command": command,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "manifest_path": str(manifest_path),
            "command": command,
        }

    def build_train_command(self, config_path: str) -> str:
        return (
            f"{self._env_exports()} && "
            f"{self._activate_conda()} && "
            f"cd \"{self.settings.unielf.project_root}\" && "
            f"{self.settings.unielf.train_entry} \"{config_path}\""
        )

    def build_inference_command(
        self,
        model_file: str,
        dataset_csv: str,
        config_path: str,
        scaler_path: str | None = None,
    ) -> str:
        scaler_part = f" --scaler \"{scaler_path}\"" if scaler_path else ""
        return (
            f"{self._env_exports()} && "
            f"{self._activate_conda()} && "
            f"cd \"{self.settings.unielf.project_root}\" && "
            f"{self.settings.unielf.inference_entry} \"{model_file}\" \"{dataset_csv}\" "
            f"--config \"{config_path}\"{scaler_part}"
        )

    def _env_exports(self) -> str:
        return (
            f"export HTTP_PROXY=\"{self.settings.proxy.http_proxy}\" && "
            f"export HTTPS_PROXY=\"{self.settings.proxy.https_proxy}\""
        )

    def _activate_conda(self) -> str:
        return (
            'source "$(conda info --base)/etc/profile.d/conda.sh" && '
            f'conda activate "{self.settings.unielf.conda_env_prefix}"'
        )


class UniMolRunner:
    """Prepare a lightweight Python launcher for Uni-Mol based workflows."""

    def __init__(self, settings: ExternalModelSettings | None = None):
        self.settings = settings or ExternalModelSettings()

    def prepare_training_job(
        self,
        dataset_csv: str,
        output_dir: str,
        target_columns: list[str] | None = None,
    ) -> dict[str, str]:
        target_dir = Path(output_dir) / "unimol"
        target_dir.mkdir(parents=True, exist_ok=True)
        script_path = target_dir / self.settings.unimol.runner_script_name
        target_columns = target_columns or ["delta_d", "delta_p", "delta_h"]

        script_path.write_text(
            self._build_training_script(dataset_csv=dataset_csv, target_columns=target_columns),
            encoding="utf-8",
        )
        manifest_path = target_dir / "train_manifest.json"
        manifest = {
            "backend": "unimol",
            "dataset_csv": dataset_csv,
            "target_columns": target_columns,
            "script_path": str(script_path),
            "command": self.build_train_command(str(script_path)),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "script_path": str(script_path),
            "manifest_path": str(manifest_path),
            "command": manifest["command"],
        }

    def build_train_command(self, script_path: str) -> str:
        return (
            f"export HTTP_PROXY=\"{self.settings.proxy.http_proxy}\" && "
            f"export HTTPS_PROXY=\"{self.settings.proxy.https_proxy}\" && "
            f"\"{self.settings.unimol.python_executable}\" \"{script_path}\""
        )

    @staticmethod
    def _build_training_script(dataset_csv: str, target_columns: list[str]) -> str:
        target_columns_repr = ", ".join(f'"{item}"' for item in target_columns)
        return "\n".join(
            [
                '"""Bootstrap script for external Uni-Mol training."""',
                "",
                "from pathlib import Path",
                "",
                "import pandas as pd",
                "",
                "DATASET = Path(" + repr(dataset_csv) + ")",
                "TARGET_COLUMNS = [" + target_columns_repr + "]",
                "",
                "frame = pd.read_csv(DATASET)",
                'print(f"Loaded dataset with {len(frame)} rows from {DATASET}")',
                'print(f"Target columns: {TARGET_COLUMNS}")',
                'print("Replace this bootstrap with the actual unimol_tools training API call.")',
                "",
            ]
        )

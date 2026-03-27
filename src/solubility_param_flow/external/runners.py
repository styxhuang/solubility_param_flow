"""Command builders for external Uni-Mol and uni-elf workflows."""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any

from solubility_param_flow.external.config import BohriumJobSettings, ExternalModelSettings


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


def _copy_input_file(source_path: str, target_dir: Path, target_name: str | None = None) -> Path:
    source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    destination = target_dir / (target_name or source.name)
    shutil.copy2(source, destination)
    return destination


def _job_group_enabled(bohrium: BohriumJobSettings) -> bool:
    return bohrium.use_job_group or bohrium.job_group_id > 0


def _resolve_job_group_name(job_name: str, bohrium: BohriumJobSettings) -> str:
    configured_name = bohrium.job_group_name.strip()
    return configured_name or job_name


def _build_job_group_create_command(job_group_name: str, bohrium: BohriumJobSettings) -> str:
    args = [
        "bohr",
        "job_group",
        "create",
        "-n",
        job_group_name,
        "-p",
        str(bohrium.project_id),
    ]
    return f"script -q -c {shlex.quote(shlex.join(args))} /dev/null"


def _job_group_extract_command(python_executable: str) -> str:
    parser = (
        "import re, sys\n"
        "text = sys.stdin.read()\n"
        "match = re.search(r'(?:job[ _-]*group(?:[ _-]*id)?)\\D+(\\d+)', text, re.I)\n"
        "if match:\n"
        "    sys.stdout.write(match.group(1))\n"
        "    raise SystemExit(0)\n"
        "numbers = re.findall(r'\\d+', text)\n"
        "if not numbers:\n"
        "    raise SystemExit('failed to parse Bohrium job group id')\n"
        "sys.stdout.write(numbers[-1])\n"
    )
    return f"{shlex.quote(python_executable)} -c {shlex.quote(parser)}"


def _build_bohrium_submit_command(
    *,
    job_dir: Path,
    job_name: str,
    remote_command: str,
    bohrium: BohriumJobSettings,
) -> tuple[str, int | None, str | None]:
    args = [
        "bohr",
        "job",
        "submit",
        "-m",
        bohrium.image,
        "-t",
        bohrium.machine_type,
        "-c",
        remote_command,
        "-p",
        ".",
        "--project_id",
        str(bohrium.project_id),
        "-n",
        job_name,
    ]
    if bohrium.result_path:
        args.extend(["-r", bohrium.result_path])
    if bohrium.max_run_time > 0:
        args.extend(["--max_run_time", str(bohrium.max_run_time)])

    job_group_id: int | None = None
    job_group_name: str | None = None
    job_group_placeholder = "__JOB_GROUP_ID__"
    if bohrium.job_group_id > 0:
        job_group_id = bohrium.job_group_id
        args.extend(["-g", str(bohrium.job_group_id)])
    elif bohrium.use_job_group:
        job_group_name = _resolve_job_group_name(job_name, bohrium)
        args.extend(["-g", job_group_placeholder])

    submit_args = shlex.join(args).replace(job_group_placeholder, '"${JOB_GROUP_ID}"')
    submit_cmd = f"script -q -c {shlex.quote(submit_args)} /dev/null"
    if job_group_name:
        create_cmd = _build_job_group_create_command(job_group_name, bohrium)
        extract_cmd = _job_group_extract_command(bohrium.python_executable)
        submit_cmd = (
            f'JOB_GROUP_ID=$({create_cmd} | {extract_cmd}) && '
            f'export JOB_GROUP_ID && {submit_cmd}'
        )
    return (
        f"cd {shlex.quote(str(job_dir))} && {submit_cmd}",
        job_group_id,
        job_group_name,
    )


class UniElfRunner:
    """Prepare config files and Bohrium submission commands for uni-elf."""

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

        local_dataset = _copy_input_file(dataset_csv, target_dir, "dataset.csv")
        config_path = target_dir / "train_config.yaml"
        config_payload = {
            "task": task,
            "data_path": local_dataset.name,
            "target_col": target_column,
            "metrics": metrics,
            "batch_size": 16,
            "max_epoch": 20,
            "tensorboard_logdir": "outputs/tsb",
            "save_dir": "outputs/ckpts",
            "tmp_save_dir": "outputs/ckpts",
        }
        config_path.write_text(_yaml_dump(config_payload) + "\n", encoding="utf-8")

        script_path = target_dir / self.settings.unielf.train_script_name
        script_path.write_text(
            self._build_train_script(config_name=config_path.name),
            encoding="utf-8",
        )

        remote_command = self.build_train_command(script_path.name)
        submit_command, job_group_id, job_group_name = _build_bohrium_submit_command(
            job_dir=target_dir,
            job_name=f"unielf-train-{target_column}",
            remote_command=remote_command,
            bohrium=self.settings.unielf.bohrium,
        )
        manifest_path = target_dir / "train_manifest.json"
        manifest = {
            "backend": "uni-elf",
            "dataset_csv": str(local_dataset),
            "config_path": str(config_path),
            "script_path": str(script_path),
            "image": self.settings.unielf.bohrium.image,
            "machine_type": self.settings.unielf.bohrium.machine_type,
            "project_id": self.settings.unielf.bohrium.project_id,
            "use_job_group": _job_group_enabled(self.settings.unielf.bohrium),
            "job_group_id": job_group_id,
            "job_group_name": job_group_name,
            "remote_command": remote_command,
            "command": submit_command,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "config_path": str(config_path),
            "script_path": str(script_path),
            "manifest_path": str(manifest_path),
            "command": submit_command,
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
        local_dataset = _copy_input_file(dataset_csv, target_dir, "inference_dataset.csv")
        local_model = _copy_input_file(model_file, target_dir, "model.pt")
        local_config = _copy_input_file(config_path, target_dir, "inference_config.yaml")
        local_scaler = (
            _copy_input_file(scaler_path, target_dir, "scaler.pkl") if scaler_path else None
        )

        script_path = target_dir / self.settings.unielf.inference_script_name
        script_path.write_text(
            self._build_inference_script(
                model_name=local_model.name,
                dataset_name=local_dataset.name,
                config_name=local_config.name,
                scaler_name=local_scaler.name if local_scaler else None,
            ),
            encoding="utf-8",
        )

        remote_command = self.build_inference_command(script_path.name)
        submit_command, job_group_id, job_group_name = _build_bohrium_submit_command(
            job_dir=target_dir,
            job_name="unielf-inference",
            remote_command=remote_command,
            bohrium=self.settings.unielf.bohrium,
        )
        manifest_path = target_dir / "inference_manifest.json"
        manifest = {
            "backend": "uni-elf",
            "dataset_csv": str(local_dataset),
            "model_file": str(local_model),
            "config_path": str(local_config),
            "scaler_path": str(local_scaler) if local_scaler else None,
            "script_path": str(script_path),
            "image": self.settings.unielf.bohrium.image,
            "machine_type": self.settings.unielf.bohrium.machine_type,
            "project_id": self.settings.unielf.bohrium.project_id,
            "use_job_group": _job_group_enabled(self.settings.unielf.bohrium),
            "job_group_id": job_group_id,
            "job_group_name": job_group_name,
            "remote_command": remote_command,
            "command": submit_command,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "script_path": str(script_path),
            "manifest_path": str(manifest_path),
            "command": submit_command,
        }

    def build_train_command(self, script_name: str) -> str:
        return (
            f"{self._env_exports()} && "
            f"{self.settings.unielf.bohrium.python_executable} {shlex.quote(script_name)}"
        )

    def build_inference_command(self, script_name: str) -> str:
        return (
            f"{self._env_exports()} && "
            f"{self.settings.unielf.bohrium.python_executable} {shlex.quote(script_name)}"
        )

    def _env_exports(self) -> str:
        return (
            f"export HTTP_PROXY={shlex.quote(self.settings.proxy.http_proxy)} && "
            f"export HTTPS_PROXY={shlex.quote(self.settings.proxy.https_proxy)}"
        )

    def _build_train_script(self, *, config_name: str) -> str:
        return "\n".join(
            [
                '"""Launch uni-elf training inside the Bohrium image."""',
                "",
                "from __future__ import annotations",
                "",
                "import subprocess",
                "from pathlib import Path",
                "",
                f'CONFIG = Path("{config_name}")',
                'OUTPUT_DIR = Path("outputs")',
                "",
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
                "subprocess.run([\"unielf-train\", str(CONFIG)], check=True)",
                "",
            ]
        )

    def _build_inference_script(
        self,
        *,
        model_name: str,
        dataset_name: str,
        config_name: str,
        scaler_name: str | None,
    ) -> str:
        scaler_part = f', "--scaler", "{scaler_name}"' if scaler_name else ""
        return "\n".join(
            [
                '"""Launch uni-elf inference inside the Bohrium image."""',
                "",
                "from __future__ import annotations",
                "",
                "import subprocess",
                "",
                "cmd = [",
                f'    "unielf-inference", "{model_name}", "{dataset_name}",',
                f'    "--config", "{config_name}"{scaler_part}',
                "]",
                "flat_cmd = []",
                "for item in cmd:",
                "    if isinstance(item, tuple):",
                "        flat_cmd.extend(item)",
                "    else:",
                "        flat_cmd.append(item)",
                "subprocess.run(flat_cmd, check=True)",
                "",
            ]
        )


class UniMolRunner:
    """Prepare Bohrium-ready Uni-Mol training scripts."""

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
        local_dataset = _copy_input_file(dataset_csv, target_dir, "dataset.csv")
        script_path = target_dir / self.settings.unimol.runner_script_name
        target_columns = target_columns or ["delta_d", "delta_p", "delta_h"]

        script_path.write_text(
            self._build_training_script(dataset_name=local_dataset.name, target_columns=target_columns),
            encoding="utf-8",
        )
        remote_command = self.build_train_command(script_path.name)
        submit_command, job_group_id, job_group_name = _build_bohrium_submit_command(
            job_dir=target_dir,
            job_name="unimol-train",
            remote_command=remote_command,
            bohrium=self.settings.unimol.bohrium,
        )
        manifest_path = target_dir / "train_manifest.json"
        manifest = {
            "backend": "unimol",
            "dataset_csv": str(local_dataset),
            "target_columns": target_columns,
            "script_path": str(script_path),
            "image": self.settings.unimol.bohrium.image,
            "machine_type": self.settings.unimol.bohrium.machine_type,
            "project_id": self.settings.unimol.bohrium.project_id,
            "use_job_group": _job_group_enabled(self.settings.unimol.bohrium),
            "job_group_id": job_group_id,
            "job_group_name": job_group_name,
            "remote_command": remote_command,
            "command": submit_command,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "script_path": str(script_path),
            "manifest_path": str(manifest_path),
            "command": submit_command,
        }

    def build_train_command(self, script_name: str) -> str:
        return (
            f"export HTTP_PROXY={shlex.quote(self.settings.proxy.http_proxy)} && "
            f"export HTTPS_PROXY={shlex.quote(self.settings.proxy.https_proxy)} && "
            f"{self.settings.unimol.bohrium.python_executable} {shlex.quote(script_name)}"
        )

    @staticmethod
    def _build_training_script(dataset_name: str, target_columns: list[str]) -> str:
        target_columns_repr = ", ".join(f'"{item}"' for item in target_columns)
        template = '''"""Train and evaluate a Uni-Mol + scalar regression model inside Bohrium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from unimol_tools import UniMolRepr

TARGET_CANDIDATE_COLS = [
    ["Dispersion  δd", "delta_d", "d"],
    ["Polarity    δp", "delta_p", "p"],
    ["Hydrogen Bonding   δd", "delta_h", "h"],
]
TARGET_LABELS = ["δd (Dispersion)", "δp (Polarity)", "δh (H-Bond)"]
TARGET_KEYS = ["dD", "dP", "dH"]
MANUAL_SCALAR_COLS = [
    "area_A2", "volume_A3", "vm_cm3_mol", "n_segments",
    "sigma_mean", "sigma_std", "sigma_skew", "sigma_kurtosis", "sigma_abs_mean", "sigma_rms",
    "sigma_m2", "sigma_m3", "sigma_m4", "sigma_m5", "sigma_m6",
    "hb_acc_m1", "hb_acc_m2", "hb_acc_m3",
    "hb_don_m1", "hb_don_m2", "hb_don_m3",
]
DEFAULT_DATASET = Path("__DATASET_NAME__")
DEFAULT_OUT_DIR = Path("outputs")
TARGET_COLUMNS = [__TARGET_COLUMNS__]
RANDOM_SEED = 42
N_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 256
SCALAR_HIDDEN_DIM = 64
DROPOUT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class MolScalarDataset(Dataset):
    def __init__(self, mol_x: np.ndarray, scalar_x: np.ndarray, y: np.ndarray):
        self.mol_x = torch.tensor(mol_x, dtype=torch.float32)
        self.scalar_x = torch.tensor(scalar_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.mol_x[idx], self.scalar_x[idx], self.y[idx]


class UniMolScalarRegressor(nn.Module):
    def __init__(self, mol_dim: int, scalar_dim: int, scalar_hidden_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_hidden_dim, scalar_hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(mol_dim + scalar_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, mol_x: torch.Tensor, scalar_x: torch.Tensor) -> torch.Tensor:
        scalar_h = self.scalar_encoder(scalar_x)
        fused = torch.cat([mol_x, scalar_h], dim=-1)
        return self.head(fused)


def run_epoch(loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    pred_list, true_list = [], []
    loss_sum, n_batch = 0.0, 0
    for mol_x, scalar_x, y in loader:
        mol_x = mol_x.to(DEVICE)
        scalar_x = scalar_x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.set_grad_enabled(is_train):
            pred = model(mol_x, scalar_x)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item()
        n_batch += 1
        pred_list.append(pred.detach().cpu().numpy())
        true_list.append(y.detach().cpu().numpy())
    return loss_sum / max(1, n_batch), np.concatenate(pred_list, axis=0), np.concatenate(true_list, axis=0)


def calc_formal_charge(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(Chem.GetFormalCharge(mol))


def prepare_dataframe(df: pd.DataFrame, scalar_cols: list[str], fill_values: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    out["SMILES"] = out["SMILES"].astype(str).str.strip()
    out = out[out["SMILES"] != ""].copy()
    out["formal_charge"] = out["SMILES"].map(calc_formal_charge)
    for col in scalar_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if fill_values is None:
        fill_values = out[scalar_cols].median(numeric_only=True).fillna(0.0)
    out[scalar_cols] = out[scalar_cols].fillna(fill_values).fillna(0.0)
    return out.reset_index(drop=True), fill_values


def resolve_target_columns(df: pd.DataFrame) -> list[str]:
    resolved = []
    for candidates, fallback in zip(TARGET_CANDIDATE_COLS, TARGET_COLUMNS):
        chosen = next((c for c in candidates if c in df.columns), None)
        if chosen is None:
            chosen = fallback if fallback in df.columns else None
        if chosen is None:
            raise ValueError(f"缺少目标列，候选列: {candidates} / fallback: {fallback}")
        resolved.append(chosen)
    return resolved


def build_representations(smiles: list[str]) -> np.ndarray:
    unique_smiles = list(dict.fromkeys(smiles))
    repr_model = UniMolRepr(data_type="molecule", remove_hs=False, use_gpu=(DEVICE == "cuda"))
    repr_out = repr_model.get_repr(unique_smiles)
    cls_repr = repr_out["cls_repr"] if isinstance(repr_out, dict) else repr_out
    mapping = dict(zip(unique_smiles, cls_repr))
    return np.stack([mapping[s] for s in smiles], axis=0).astype(np.float32)


def train_and_save(train_df_raw: pd.DataFrame, test_df_raw: pd.DataFrame, out_dir: Path, model_dir: Path) -> None:
    target_cols = resolve_target_columns(pd.concat([train_df_raw, test_df_raw], axis=0, ignore_index=True))
    scalar_cols = ["formal_charge"] + [c for c in MANUAL_SCALAR_COLS if c in train_df_raw.columns or c in test_df_raw.columns]
    train_df, fill_values = prepare_dataframe(train_df_raw, scalar_cols)
    test_df, _ = prepare_dataframe(test_df_raw, scalar_cols, fill_values=fill_values)
    for tc in target_cols:
        train_df[tc] = pd.to_numeric(train_df.get(tc, np.nan), errors="coerce")
        test_df[tc] = pd.to_numeric(test_df.get(tc, np.nan), errors="coerce")

    train_smiles = train_df["SMILES"].tolist()
    train_mol = build_representations(train_smiles)
    train_scalar = train_df[scalar_cols].to_numpy(dtype=np.float32)
    cv_rows: list[dict] = []
    metric_rows: list[dict] = []
    saved_model_paths = []

    for target_col, target_key, target_label in zip(target_cols, TARGET_KEYS, TARGET_LABELS):
        target_mask = train_df[target_col].notna().to_numpy()
        y_all = train_df.loc[target_mask, target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        mol_all = train_mol[target_mask]
        scalar_all = train_scalar[target_mask]
        if len(y_all) < N_FOLDS:
            raise ValueError(f"{target_label} 可用样本太少: {len(y_all)}")

        fold_r2 = []
        fold_mae = []
        fold_rmse = []
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        for fold, (idx_tr, idx_va) in enumerate(kf.split(np.arange(len(y_all))), start=1):
            tr_mol, va_mol = mol_all[idx_tr], mol_all[idx_va]
            tr_scalar, va_scalar = scalar_all[idx_tr], scalar_all[idx_va]
            tr_y, va_y = y_all[idx_tr], y_all[idx_va]
            scalar_scaler = StandardScaler()
            y_scaler = StandardScaler()
            tr_scalar_s = scalar_scaler.fit_transform(tr_scalar).astype(np.float32)
            va_scalar_s = scalar_scaler.transform(va_scalar).astype(np.float32)
            tr_y_s = y_scaler.fit_transform(tr_y).astype(np.float32)
            va_y_s = y_scaler.transform(va_y).astype(np.float32)

            tr_loader = DataLoader(MolScalarDataset(tr_mol, tr_scalar_s, tr_y_s), batch_size=BATCH_SIZE, shuffle=True)
            va_loader = DataLoader(MolScalarDataset(va_mol, va_scalar_s, va_y_s), batch_size=BATCH_SIZE, shuffle=False)
            model = UniMolScalarRegressor(
                mol_dim=mol_all.shape[1],
                scalar_dim=scalar_all.shape[1],
                scalar_hidden_dim=SCALAR_HIDDEN_DIM,
                hidden_dim=HIDDEN_DIM,
                dropout=DROPOUT,
            ).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            best_state = None
            best_r2 = -1e9
            for _ in range(EPOCHS):
                run_epoch(tr_loader, model, criterion, optimizer=optimizer)
                _, va_pred_s, va_true_s = run_epoch(va_loader, model, criterion, optimizer=None)
                va_pred = y_scaler.inverse_transform(va_pred_s).reshape(-1)
                va_true = y_scaler.inverse_transform(va_true_s).reshape(-1)
                curr_r2 = r2_score(va_true, va_pred)
                if curr_r2 > best_r2:
                    best_r2 = curr_r2
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            _, va_pred_s, va_true_s = run_epoch(va_loader, model, criterion, optimizer=None)
            va_pred = y_scaler.inverse_transform(va_pred_s).reshape(-1)
            va_true = y_scaler.inverse_transform(va_true_s).reshape(-1)
            fold_r2.append(float(r2_score(va_true, va_pred)))
            fold_mae.append(float(mean_absolute_error(va_true, va_pred)))
            fold_rmse.append(float(np.sqrt(mean_squared_error(va_true, va_pred))))
            cv_rows.append({"target": target_label, "fold": fold, "r2": fold_r2[-1], "mae": fold_mae[-1], "rmse": fold_rmse[-1]})

        full_scalar_scaler = StandardScaler()
        full_y_scaler = StandardScaler()
        scalar_s = full_scalar_scaler.fit_transform(scalar_all).astype(np.float32)
        y_s = full_y_scaler.fit_transform(y_all).astype(np.float32)
        full_loader = DataLoader(MolScalarDataset(mol_all, scalar_s, y_s), batch_size=BATCH_SIZE, shuffle=True)
        model = UniMolScalarRegressor(
            mol_dim=mol_all.shape[1],
            scalar_dim=scalar_all.shape[1],
            scalar_hidden_dim=SCALAR_HIDDEN_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        ).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(EPOCHS):
            run_epoch(full_loader, model, criterion, optimizer=optimizer)

        payload = {
            "target_col": target_col,
            "target_key": target_key,
            "target_label": target_label,
            "scalar_cols": scalar_cols,
            "fill_values": {k: float(v) for k, v in fill_values.to_dict().items()},
            "mol_dim": int(mol_all.shape[1]),
            "scalar_dim": int(scalar_all.shape[1]),
            "scalar_hidden_dim": SCALAR_HIDDEN_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "scalar_mean": full_scalar_scaler.mean_.copy(),
            "scalar_scale": np.where(full_scalar_scaler.scale_ != 0, full_scalar_scaler.scale_, 1.0),
            "y_mean": full_y_scaler.mean_.copy(),
            "y_scale": np.where(full_y_scaler.scale_ != 0, full_y_scaler.scale_, 1.0),
        }
        save_path = model_dir / f"{target_key}.pt"
        torch.save(payload, save_path)
        saved_model_paths.append(str(save_path))
        metric_rows.append(
            {
                "target": target_label,
                "r2_mean": float(np.mean(fold_r2)),
                "r2_std": float(np.std(fold_r2)),
                "mae_mean": float(np.mean(fold_mae)),
                "rmse_mean": float(np.mean(fold_rmse)),
            }
        )

    pd.DataFrame(cv_rows).to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "cv_summary.csv", index=False)
    (out_dir / "saved_model_paths.json").write_text(json.dumps(saved_model_paths, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_payloads(model_dir: Path) -> list[dict]:
    payloads = []
    for target_key in TARGET_KEYS:
        p = model_dir / f"{target_key}.pt"
        if not p.exists():
            raise FileNotFoundError(f"模型不存在: {p}")
        payloads.append(torch.load(p, map_location="cpu", weights_only=False))
    return payloads


def predict_with_saved_models(predict_df_raw: pd.DataFrame, out_dir: Path, model_dir: Path) -> None:
    payloads = load_model_payloads(model_dir)
    scalar_cols = payloads[0]["scalar_cols"]
    fill_values = pd.Series(payloads[0]["fill_values"])
    df, _ = prepare_dataframe(predict_df_raw, scalar_cols, fill_values=fill_values)
    mol = build_representations(df["SMILES"].tolist())
    scalar = df[scalar_cols].to_numpy(dtype=np.float32)
    mol_t = torch.tensor(mol, dtype=torch.float32, device=DEVICE)
    output_df = pd.DataFrame({"SMILES": df["SMILES"].values})

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=150, facecolor="white")
    for i, (payload, target_key) in enumerate(zip(payloads, TARGET_KEYS)):
        target_col = payload["target_col"]
        target_label = payload["target_label"]
        scalar_scaled = ((scalar - payload["scalar_mean"]) / payload["scalar_scale"]).astype(np.float32)
        scalar_t = torch.tensor(scalar_scaled, dtype=torch.float32, device=DEVICE)
        model = UniMolScalarRegressor(
            mol_dim=int(payload["mol_dim"]),
            scalar_dim=int(payload["scalar_dim"]),
            scalar_hidden_dim=int(payload["scalar_hidden_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
        ).to(DEVICE)
        model.load_state_dict(payload["model_state"])
        model.eval()
        with torch.no_grad():
            pred_scaled = model(mol_t, scalar_t).cpu().numpy().reshape(-1, 1)
        pred = pred_scaled * payload["y_scale"] + payload["y_mean"]
        pred = pred.reshape(-1)
        output_df[target_key] = pred

        ax = axes[i]
        ax.set_title(target_label)
        truth = pd.to_numeric(predict_df_raw.get(target_col, pd.Series(np.nan, index=predict_df_raw.index)), errors="coerce")
        if len(truth) == len(df) and truth.notna().sum() > 2:
            valid = truth.notna().to_numpy()
            y_true = truth.to_numpy()[valid].astype(np.float32)
            y_pred = pred[valid]
            r2 = r2_score(y_true, y_pred)
            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            pad = (hi - lo) * 0.06 if hi > lo else 1.0
            lo, hi = lo - pad, hi + pad
            ax.scatter(y_true, y_pred, s=15, alpha=0.7, c="#2c3e50")
            ax.plot([lo, hi], [lo, hi], "--", color="#c0392b", linewidth=1.1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.grid(linestyle="--", alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
            ax.text(0.05, 0.95, f"R²={r2:.3f}", transform=ax.transAxes, ha="left", va="top")
        else:
            ax.text(0.5, 0.5, "No target values", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    pred_csv = out_dir / "prediction.csv"
    output_df.to_csv(pred_csv, index=False)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_scatter.png", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def split_dataset(frame: pd.DataFrame, target_cols: list[str], test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = frame[frame[target_cols].notna().all(axis=1)].copy()
    if len(labeled) < max(10, N_FOLDS * 2):
        raise ValueError(f"用于 Uni-Mol 训练的完整带标签样本过少: {len(labeled)}")
    train_df, test_df = train_test_split(
        labeled,
        test_size=test_fraction,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict", "all"], default="all")
    ap.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--predict-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    args = ap.parse_args()

    frame = pd.read_csv(args.dataset_csv)
    target_cols = resolve_target_columns(frame)
    train_df, test_df = split_dataset(frame, target_cols, args.test_fraction)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "all"):
        train_and_save(train_df, test_df, out_dir, model_dir)
    if args.mode in ("predict", "all"):
        predict_df = pd.read_csv(args.predict_csv) if args.predict_csv else test_df.copy()
        predict_with_saved_models(predict_df, out_dir, model_dir)


if __name__ == "__main__":
    main()
'''
        return (
            template.replace("__DATASET_NAME__", dataset_name).replace(
                "__TARGET_COLUMNS__", target_columns_repr
            )
        )

"""Execution settings for quantum and COSMO-RS backends."""

from typing import Literal

from pydantic import BaseModel, Field


class OrcaExecutionConfig(BaseModel):
    """Configuration for local or remote ORCA execution."""

    mode: Literal["dry-run", "local", "remote"] = "dry-run"
    orca_binary: str = "/root/orca600/orca"
    allow_run_as_root: bool = True
    nprocs: int = 2
    remote_image: str = "registry.dp.tech/dptech/orca:6.0.0"
    remote_machine_type: str = "c2_m8_cpu"
    remote_project_id: int = 3824565


class OpenCosmoExecutionConfig(BaseModel):
    """Configuration for OpenCOSMO-RS execution."""

    mode: Literal["dry-run", "local"] = "dry-run"
    python_executable: str = "python"
    reference_url: str = "https://github.com/TUHH-TVT/openCOSMO-RS_py"


class WorkflowExecutionSettings(BaseModel):
    """Top-level settings used by the CSV-to-HSP workflow pipeline."""

    smiles_column: str = "smiles"
    name_column: str = "name"
    orca: OrcaExecutionConfig = Field(default_factory=OrcaExecutionConfig)
    opencosmo: OpenCosmoExecutionConfig = Field(default_factory=OpenCosmoExecutionConfig)

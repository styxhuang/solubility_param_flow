"""Pydantic models for the SMILES-to-HSP workflow."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class MoleculeRecord(BaseModel):
    """Normalized molecule metadata extracted from CSV input."""

    molecule_id: str
    row_index: int
    input_smiles: str
    canonical_smiles: str
    name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrcaDryRunResult(BaseModel):
    """Prepared ORCA job metadata without actual execution."""

    mode: Literal["dry-run"] = "dry-run"
    status: Literal["prepared", "failed"] = "prepared"
    workdir: str
    xyz_path: str
    opt_input_path: str
    sp_input_path: str
    cosmo_input_path: str
    output_path: str
    command: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CosmoRsResult(BaseModel):
    """Mock OpenCOSMO-RS result used for dry-run integration tests."""

    mode: Literal["dry-run"] = "dry-run"
    status: Literal["prepared", "failed"] = "prepared"
    hsp_source: Literal["mock-opencosmo"] = "mock-opencosmo"
    delta_d: float
    delta_p: float
    delta_h: float
    descriptors: Dict[str, float] = Field(default_factory=dict)
    result_path: str


class PipelineRecord(BaseModel):
    """Final artifact describing one full pipeline run."""

    molecule: MoleculeRecord
    status: Literal["success", "invalid_smiles", "failed"]
    error_message: Optional[str] = None
    orca: Optional[OrcaDryRunResult] = None
    cosmo: Optional[CosmoRsResult] = None

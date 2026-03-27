"""Configuration models for external model environments."""

from pydantic import BaseModel, Field


class ProxySettings(BaseModel):
    """HTTP proxy settings used for external environments."""

    http_proxy: str = "http://ga.dp.tech:8118"
    https_proxy: str = "http://ga.dp.tech:8118"


class UniElfSettings(BaseModel):
    """External project settings for uni-elf."""

    project_root: str = "/root/software/uni-elf"
    conda_env_prefix: str = "/root/software/uni-elf/.conda-env"
    train_entry: str = "unielf-train"
    inference_entry: str = "unielf-inference"
    python_version: str = "3.10"
    bootstrap_script: str = "scripts/setup_uni_elf_env.sh"


class UniMolSettings(BaseModel):
    """External runner settings for Uni-Mol tooling."""

    python_executable: str = "/root/software/uni-elf/.conda-env/bin/python"
    package_name: str = "unimol_tools"
    runner_script_name: str = "run_unimol.py"


class ExternalModelSettings(BaseModel):
    """Top-level settings for external model orchestration."""

    proxy: ProxySettings = Field(default_factory=ProxySettings)
    unielf: UniElfSettings = Field(default_factory=UniElfSettings)
    unimol: UniMolSettings = Field(default_factory=UniMolSettings)

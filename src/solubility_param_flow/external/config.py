"""Configuration models for external model environments."""

from pydantic import BaseModel, Field


class ProxySettings(BaseModel):
    """HTTP proxy settings used for external environments."""

    http_proxy: str = "http://ga.dp.tech:8118"
    https_proxy: str = "http://ga.dp.tech:8118"


class BohriumJobSettings(BaseModel):
    """Bohrium submission settings for one external backend."""

    image: str = "registry.dp.tech/dptech/dp/native/prod-13375/unielf:v1.21.0-manual"
    machine_type: str = "c4_m8_cpu"
    project_id: int = 929872
    python_executable: str = "python"
    result_path: str = ""
    max_run_time: int = 0
    use_job_group: bool = False
    job_group_name: str = ""
    job_group_id: int = 0


class UniElfSettings(BaseModel):
    """External project and Bohrium settings for uni-elf."""

    project_root: str = "/root/software/uni-elf"
    conda_env_prefix: str = "/root/software/uni-elf/.conda-env"
    train_entry: str = "unielf-train"
    inference_entry: str = "unielf-inference"
    python_version: str = "3.10"
    bootstrap_script: str = "scripts/setup_uni_elf_env.sh"
    train_script_name: str = "run_unielf_train.py"
    inference_script_name: str = "run_unielf_inference.py"
    bohrium: BohriumJobSettings = Field(default_factory=BohriumJobSettings)


class UniMolSettings(BaseModel):
    """External runner and Bohrium settings for Uni-Mol tooling."""

    python_executable: str = "python"
    package_name: str = "unimol_tools"
    runner_script_name: str = "run_unimol.py"
    bohrium: BohriumJobSettings = Field(
        default_factory=lambda: BohriumJobSettings(
            machine_type="c16_m62_1 * NVIDIA T4",
            use_job_group=True,
            job_group_name="unimol-train",
        )
    )


class ExternalModelSettings(BaseModel):
    """Top-level settings for external model orchestration."""

    proxy: ProxySettings = Field(default_factory=ProxySettings)
    unielf: UniElfSettings = Field(default_factory=UniElfSettings)
    unimol: UniMolSettings = Field(default_factory=UniMolSettings)

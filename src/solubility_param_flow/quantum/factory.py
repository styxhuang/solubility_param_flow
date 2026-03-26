"""Factory helpers for workflow backends."""

from solubility_param_flow.quantum.dry_run import DryRunOpenCosmoRunner, DryRunOrcaRunner
from solubility_param_flow.schemas import WorkflowExecutionSettings


class WorkflowBackendFactory:
    """Create the currently supported workflow backends."""

    @staticmethod
    def create(settings: WorkflowExecutionSettings) -> tuple[DryRunOrcaRunner, DryRunOpenCosmoRunner]:
        if settings.orca.mode != "dry-run":
            raise NotImplementedError(
                "This phase only supports dry-run ORCA orchestration. "
                "Local and remote execution hooks are configured but not enabled yet."
            )
        if settings.opencosmo.mode != "dry-run":
            raise NotImplementedError(
                "This phase only supports dry-run OpenCOSMO-RS orchestration. "
                "Local execution hooks will be enabled in a later stage."
            )

        return (
            DryRunOrcaRunner(
                orca_binary=settings.orca.orca_binary,
                allow_run_as_root=settings.orca.allow_run_as_root,
                nprocs=settings.orca.nprocs,
                remote_image=settings.orca.remote_image,
                remote_machine_type=settings.orca.remote_machine_type,
                remote_project_id=settings.orca.remote_project_id,
            ),
            DryRunOpenCosmoRunner(
                python_executable=settings.opencosmo.python_executable,
                reference_url=settings.opencosmo.reference_url,
            ),
        )

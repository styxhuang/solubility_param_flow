"""CLI interface for solubility parameter workflow."""

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from solubility_param_flow.schemas import WorkflowExecutionSettings
from solubility_param_flow.workflow import SmilesToOrcaWorkflow

app = typer.Typer(help="Solubility Parameter Calculation Workflow")
console = Console()


@app.command()
def calculate_hsp(smiles: str, name: str = ""):
    """Calculate HSP for a molecule."""
    from solubility_param_flow.core.hsp_calculator import HSPCalculator

    calc = HSPCalculator()
    hsp = calc.calculate_from_smiles(smiles, name)
    
    table = Table(title=f"HSP Results for {name or smiles}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("δD (Dispersion)", f"{hsp.delta_d:.2f}")
    table.add_row("δP (Polar)", f"{hsp.delta_p:.2f}")
    table.add_row("δH (H-bond)", f"{hsp.delta_h:.2f}")
    table.add_row("δTotal", f"{hsp.delta_total:.2f}")
    
    console.print(table)


@app.command()
def descriptors(smiles: str):
    """Calculate molecular descriptors."""
    from solubility_param_flow.descriptors.molecular_descriptor import DescriptorCalculator

    calc = DescriptorCalculator()
    desc = calc.calculate(smiles)
    
    table = Table(title=f"Descriptors for {smiles}")
    table.add_column("Descriptor", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Molecular Weight", f"{desc.mw:.2f}")
    table.add_row("LogP", f"{desc.logp:.2f}")
    table.add_row("TPSA", f"{desc.tpsa:.2f}")
    table.add_row("HBD", str(desc.hbd))
    table.add_row("HBA", str(desc.hba))
    table.add_row("Rotatable Bonds", str(desc.rotatable_bonds))
    
    console.print(table)


@app.command()
def prepare_dataset(
    csv_path: str,
    output_dir: str = "artifacts/hsp_dry_run",
    smiles_column: str = "smiles",
    name_column: str = "name",
    orca_binary: str = "/root/orca600/orca",
):
    """Run the dry-run CSV -> ORCA/OpenCOSMO-RS -> HSP workflow."""
    from solubility_param_flow.pipelines.smiles_pipeline import SmilesToHSPDryRunPipeline

    settings = WorkflowExecutionSettings(
        smiles_column=smiles_column,
        name_column=name_column,
    )
    settings.orca.orca_binary = orca_binary
    pipeline = SmilesToHSPDryRunPipeline(settings=settings)
    result_frame = pipeline.run(csv_path, output_dir)

    summary = Table(title="Dry-run HSP Workflow Summary", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Input CSV", csv_path)
    summary.add_row("Output Directory", output_dir)
    summary.add_row("Total Rows", str(len(result_frame)))
    summary.add_row("Success", str((result_frame["status"] == "success").sum()))
    summary.add_row("Invalid SMILES", str((result_frame["status"] == "invalid_smiles").sum()))
    summary.add_row("Failed", str((result_frame["status"] == "failed").sum()))
    console.print(summary)
    console.print(
        f"Results saved to [bold]{output_dir}/results/hsp_workflow_results.csv[/bold]"
    )


@app.command("prepare-orca-jobs")
def prepare_orca_jobs(
    csv_path: str,
    output_dir: str = "artifacts/orca_jobs",
    seed: int = 42,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "XTB2",
    basis: str = "",
    sp_functional: str = "BP86",
    sp_basis: str = "def2-TZVP",
    nproc: int = 4,
    maxcore: int = 2000,
    epsilon: float = 1.0e8,
    no_opt: bool = False,
    orca_binary: str = "orca",
    start: int = 1,
    end: int = 0,
):
    """Generate row-wise ORCA job directories from a CSV file."""
    from solubility_param_flow.quantum.orca_job_builder import build_orca_jobs_from_csv

    summary_data = build_orca_jobs_from_csv(
        csv_path=csv_path,
        output_dir=output_dir,
        seed=seed,
        charge=charge,
        multiplicity=multiplicity,
        functional=functional,
        basis=basis,
        sp_functional=sp_functional,
        sp_basis=sp_basis,
        nproc=nproc,
        maxcore=maxcore,
        epsilon=epsilon,
        no_opt=no_opt,
        orca_binary=orca_binary,
        start=start,
        end=end,
    )

    summary = Table(title="ORCA Job Preparation", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Input CSV", csv_path)
    summary.add_row("Output Directory", output_dir)
    summary.add_row("Total Rows", str(summary_data.total_rows))
    summary.add_row("Generated Jobs", str(summary_data.generated_jobs))
    summary.add_row("Missing SMILES", str(summary_data.skipped_missing_smiles))
    summary.add_row("Invalid SMILES", str(summary_data.skipped_invalid_smiles))
    summary.add_row("Run Script", summary_data.run_script)
    console.print(summary)


@app.command("submit-orca-jobs")
def submit_orca_jobs_cli(
    jobs_dir: str,
    project_id: int = 929872,
    image: str = "registry.dp.tech/dptech/prod-13629/orca-xtb:6.0.0_6.7.1",
    machine_type: str = "c4_m8_cpu",
    name_prefix: str = "orca",
    manifest_path: str = "",
    start: int = 1,
    end: int = 0,
    orca_binary: str = "/root/orca600/orca",
    xtb_binary: str = "/root/xtb-dist/bin/xtb",
    allow_run_as_root: bool = True,
    max_run_time: int = 0,
    result_path: str = "",
):
    """Submit prepared `rN` ORCA job folders to Bohrium."""
    from solubility_param_flow.quantum.orca_bohrium import submit_orca_jobs

    summary_data = submit_orca_jobs(
        jobs_dir=jobs_dir,
        project_id=project_id,
        image=image,
        machine_type=machine_type,
        name_prefix=name_prefix,
        manifest_path=manifest_path or None,
        start=start,
        end=end,
        orca_binary=orca_binary,
        xtb_binary=xtb_binary,
        allow_run_as_root=allow_run_as_root,
        max_run_time=max_run_time,
        result_path=result_path,
    )

    summary = Table(title="Bohrium ORCA Submission", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Jobs Directory", jobs_dir)
    summary.add_row("Project ID", str(project_id))
    summary.add_row("Image", image)
    summary.add_row("Machine Type", machine_type)
    summary.add_row("Submitted", str(summary_data.submitted_count))
    summary.add_row("Failed", str(summary_data.failed_count))
    summary.add_row("Manifest", summary_data.manifest_path)
    console.print(summary)


@app.command("submit-orca-csv")
def submit_orca_csv(
    csv_path: str,
    jobs_dir: str = "artifacts/orca_jobs",
    project_id: int = 929872,
    image: str = "registry.dp.tech/dptech/prod-13629/orca-xtb:6.0.0_6.7.1",
    machine_type: str = "c4_m8_cpu",
    name_prefix: str = "orca",
    manifest_path: str = "",
    seed: int = 42,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "XTB2",
    basis: str = "",
    sp_functional: str = "BP86",
    sp_basis: str = "def2-TZVP",
    nproc: int = 4,
    maxcore: int = 2000,
    epsilon: float = 1.0e8,
    no_opt: bool = False,
    orca_binary: str = "/root/orca600/orca",
    xtb_binary: str = "/root/xtb-dist/bin/xtb",
    allow_run_as_root: bool = True,
    start: int = 1,
    end: int = 0,
    max_run_time: int = 0,
    result_path: str = "",
):
    """Build ORCA jobs from CSV and submit them to Bohrium."""
    from solubility_param_flow.quantum.orca_bohrium import submit_orca_jobs
    from solubility_param_flow.quantum.orca_job_builder import build_orca_jobs_from_csv

    build_summary = build_orca_jobs_from_csv(
        csv_path=csv_path,
        output_dir=jobs_dir,
        seed=seed,
        charge=charge,
        multiplicity=multiplicity,
        functional=functional,
        basis=basis,
        sp_functional=sp_functional,
        sp_basis=sp_basis,
        nproc=nproc,
        maxcore=maxcore,
        epsilon=epsilon,
        no_opt=no_opt,
        orca_binary=orca_binary,
        start=start,
        end=end,
    )
    submit_summary = submit_orca_jobs(
        jobs_dir=jobs_dir,
        project_id=project_id,
        image=image,
        machine_type=machine_type,
        name_prefix=name_prefix,
        manifest_path=manifest_path or None,
        start=start,
        end=end,
        orca_binary=orca_binary,
        xtb_binary=xtb_binary,
        allow_run_as_root=allow_run_as_root,
        max_run_time=max_run_time,
        result_path=result_path,
    )

    summary = Table(title="CSV to Bohrium ORCA", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Input CSV", csv_path)
    summary.add_row("Jobs Directory", jobs_dir)
    summary.add_row("Prepared Jobs", str(build_summary.generated_jobs))
    summary.add_row("Submitted Jobs", str(submit_summary.submitted_count))
    summary.add_row("Failed Submit", str(submit_summary.failed_count))
    summary.add_row("Manifest", submit_summary.manifest_path)
    console.print(summary)


@app.command("download-orca-results")
def download_orca_results(
    manifest_path: str,
    output_dir: str = "",
    completed_only: bool = False,
):
    """Download Bohrium job outputs using a submission manifest."""
    from solubility_param_flow.quantum.orca_bohrium import download_job_results

    summary_data = download_job_results(
        manifest_path=manifest_path,
        output_dir=output_dir or None,
        completed_only=completed_only,
    )

    summary = Table(title="Bohrium Result Download", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Manifest", manifest_path)
    summary.add_row("Output Directory", summary_data.output_dir)
    summary.add_row("Requested Jobs", str(summary_data.requested_count))
    summary.add_row("Downloaded Entries", str(summary_data.downloaded_count))
    console.print(summary)


@app.command("stage-orca-results")
def stage_orca_results(
    manifest_path: str,
    download_dir: str,
    jobs_dir: str = "",
):
    """Copy downloaded Bohrium outputs back into local `rN` folders."""
    from solubility_param_flow.quantum.orca_bohrium import stage_downloaded_results

    copied = stage_downloaded_results(
        manifest_path=manifest_path,
        download_dir=download_dir,
        jobs_dir=jobs_dir or None,
    )
    console.print(f"Copied [bold]{copied}[/bold] files into local job folders.")


@app.command("post-orca")
def post_orca(
    csv_path: str,
    jobs_dir: str,
    output_csv: str = "",
    backup: bool = False,
    start: int = 1,
    end: int = 0,
    scale_d: float = 1.0,
    scale_p: float = 1.0,
    scale_h: float = 1.0,
    sigma_width: float = 0.010,
    polar_mode: str = "weighted",
    enable_weak_hbond_correction: bool = False,
    fit_exp: bool = False,
    fit_targets: str = "d,p,h",
    fit_count: int = 0,
    fit_offset: int = 0,
    exp_col_d: int = 3,
    exp_col_p: int = 4,
    exp_col_h: int = 5,
):
    """Post-process ORCA `.cpcm` outputs and write d/p/h/Vm back to CSV."""
    from solubility_param_flow.quantum.orca_postprocess import postprocess_orca_jobs

    summary_data = postprocess_orca_jobs(
        csv_path=csv_path,
        jobs_dir=jobs_dir,
        output_csv=output_csv or None,
        backup=backup,
        start=start,
        end=end,
        scale_d=scale_d,
        scale_p=scale_p,
        scale_h=scale_h,
        sigma_width=sigma_width,
        polar_mode=polar_mode,
        enable_weak_hbond_correction=enable_weak_hbond_correction,
        fit_exp=fit_exp,
        fit_targets=fit_targets,
        fit_count=fit_count,
        fit_offset=fit_offset,
        exp_col_d=exp_col_d,
        exp_col_p=exp_col_p,
        exp_col_h=exp_col_h,
    )

    summary = Table(title="ORCA Post-processing", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Input CSV", csv_path)
    summary.add_row("Jobs Directory", jobs_dir)
    summary.add_row("Output CSV", summary_data.output_csv)
    summary.add_row("Success", str(summary_data.success_count))
    summary.add_row("Missing CPCM", str(summary_data.missing_count))
    summary.add_row("Failed", str(summary_data.failed_count))
    console.print(summary)


@app.command("export-sigma")
def export_sigma(
    jobs_dir: str,
    output_path: str = "artifacts/orca_jobs/descriptors.csv",
    csv_path: str = "",
):
    """Export sigma descriptors from ORCA `.cpcm` outputs."""
    from solubility_param_flow.quantum.sigma_descriptors import export_sigma_descriptors

    summary_data = export_sigma_descriptors(
        jobs_dir=jobs_dir,
        csv_path=csv_path or None,
        output_path=output_path,
    )

    summary = Table(title="Sigma Descriptor Export", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Jobs Directory", jobs_dir)
    summary.add_row("Output CSV", summary_data.output_path)
    summary.add_row("Success", str(summary_data.ok_count))
    summary.add_row("Missing CPCM", str(summary_data.missing_count))
    summary.add_row("Failed", str(summary_data.failed_count))
    console.print(summary)


@app.command()
def train_ml(
    dataset_path: str,
    output_dir: str = "artifacts/ml_baseline",
    feature_sets: str = "rdkit,cosmo,combined",
):
    """Train traditional ML baselines on HSP targets."""
    from solubility_param_flow.models.hsp_trainer import TraditionalMLTrainer

    trainer = TraditionalMLTrainer()
    selected_feature_sets = [item.strip() for item in feature_sets.split(",") if item.strip()]
    metrics = trainer.run_benchmark(
        dataset_path=dataset_path,
        output_dir=output_dir,
        feature_sets=selected_feature_sets,
    )

    summary = Table(title="Traditional ML Benchmark", box=box.SIMPLE_HEAVY)
    summary.add_column("Feature Set", style="cyan")
    summary.add_column("Model", style="green")
    summary.add_column("MAE", style="magenta")
    summary.add_column("RMSE", style="magenta")
    summary.add_column("R2", style="magenta")

    for _, row in metrics.sort_values(["feature_set", "mae"]).iterrows():
        summary.add_row(
            str(row["feature_set"]),
            str(row["model"]),
            f"{row['mae']:.4f}",
            f"{row['rmse']:.4f}",
            f"{row['r2']:.4f}",
        )

    console.print(summary)
    console.print(f"Metrics saved to [bold]{output_dir}/metrics_summary.csv[/bold]")
    console.print(f"Plots saved under [bold]{output_dir}[/bold]")


@app.command()
def prepare_external_models(
    dataset_path: str,
    output_dir: str = "artifacts/external_models",
    unielf_target: str = "delta_d",
    submit: bool = False,
):
    """Prepare Uni-Mol and uni-elf command manifests."""
    from solubility_param_flow.external.config import ExternalModelSettings
    from solubility_param_flow.external.runners import UniElfRunner, UniMolRunner

    settings = ExternalModelSettings()
    unielf_runner = UniElfRunner(settings=settings)
    unimol_runner = UniMolRunner(settings=settings)

    unielf_artifacts = unielf_runner.prepare_training_job(
        dataset_csv=dataset_path,
        output_dir=output_dir,
        target_column=unielf_target,
    )
    unimol_artifacts = unimol_runner.prepare_training_job(
        dataset_csv=dataset_path,
        output_dir=output_dir,
    )

    summary = Table(title="External Model Artifacts", box=box.SIMPLE_HEAVY)
    summary.add_column("Backend", style="cyan")
    summary.add_column("Artifact", style="magenta")
    summary.add_row("uni-elf", unielf_artifacts["script_path"])
    summary.add_row("uni-elf", unielf_artifacts["config_path"])
    summary.add_row("uni-elf", unielf_artifacts["manifest_path"])
    summary.add_row("uni-elf", settings.unielf.bohrium.machine_type)
    summary.add_row("uni-elf", str(settings.unielf.bohrium.use_job_group))
    summary.add_row("Uni-Mol", unimol_artifacts["script_path"])
    summary.add_row("Uni-Mol", unimol_artifacts["manifest_path"])
    summary.add_row("Uni-Mol", settings.unimol.bohrium.machine_type)
    summary.add_row("Uni-Mol", str(settings.unimol.bohrium.use_job_group))
    console.print(summary)

    console.print("\n[bold]uni-elf Bohrium submit command[/bold]")
    console.print(unielf_artifacts["command"])
    console.print("\n[bold]Uni-Mol Bohrium submit command[/bold]")
    console.print(unimol_artifacts["command"])
    if submit:
        import subprocess

        subprocess.run(["bash", "-lc", unielf_artifacts["command"]], check=True)
        subprocess.run(["bash", "-lc", unimol_artifacts["command"]], check=True)


@app.command()
def prepare_unielf_inference(
    dataset_path: str,
    model_file: str,
    config_path: str,
    scaler_path: str = "",
    output_dir: str = "artifacts/external_models",
    submit: bool = False,
):
    """Prepare a uni-elf inference command manifest."""
    from solubility_param_flow.external.config import ExternalModelSettings
    from solubility_param_flow.external.runners import UniElfRunner

    settings = ExternalModelSettings()
    runner = UniElfRunner(settings=settings)
    artifacts = runner.prepare_inference_job(
        dataset_csv=dataset_path,
        model_file=model_file,
        output_dir=output_dir,
        config_path=config_path,
        scaler_path=scaler_path or None,
    )

    summary = Table(title="uni-elf Inference Manifest", box=box.SIMPLE_HEAVY)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Manifest", artifacts["manifest_path"])
    summary.add_row("Config", config_path)
    summary.add_row("Model", model_file)
    summary.add_row("Dataset", dataset_path)
    summary.add_row("Script", artifacts["script_path"])
    summary.add_row("Machine", settings.unielf.bohrium.machine_type)
    summary.add_row("Use Job Group", str(settings.unielf.bohrium.use_job_group))
    console.print(summary)
    console.print("\n[bold]uni-elf Bohrium inference submit command[/bold]")
    console.print(artifacts["command"])
    if submit:
        import subprocess

        subprocess.run(["bash", "-lc", artifacts["command"]], check=True)


@app.command()
def show_unielf_setup_script(script_path: str = "scripts/setup_uni_elf_env.sh"):
    """Show the uni-elf environment bootstrap script path."""
    resolved = Path(script_path)
    console.print(f"uni-elf setup script: [bold]{resolved}[/bold]")
    if resolved.exists():
        console.print("Run it with:")
        console.print(f"bash {resolved}")
    else:
        console.print("[bold red]Setup script not found[/bold red]")


@app.command()
def predict(solute: str, solvent: str):
    """Predict solubility of solute in solvent."""
    from solubility_param_flow.core.hsp_calculator import HSPCalculator

    calc = HSPCalculator()
    solute_hsp = calc.calculate_from_smiles(solute, "solute")
    solvent_hsp = calc.calculate_from_smiles(solvent, "solvent")
    
    solubility = calc.predict_solubility(solute_hsp, solvent_hsp)
    
    console.print(f"\n[bold green]Predicted Solubility: {solubility:.3f}[/bold green]")
    console.print(f"HSP Distance: {solute_hsp.distance_to(solvent_hsp):.2f}")


@app.command()
def submit_orca(smiles: str, name: str = "molecule", project_id: int = 929872):
    """Submit ORCA job from SMILES to Bohrium."""
    console.print(f"[bold blue]Starting workflow for {name}...[/bold blue]")
    console.print(f"SMILES: {smiles}")
    
    workflow = SmilesToOrcaWorkflow(project_id=project_id)
    job_id = workflow.run_workflow(smiles, name)
    
    if job_id:
        console.print(f"\n[bold green]✓ Job submitted successfully![/bold green]")
        console.print(f"Job ID: {job_id}")
        console.print(f"Monitor: bohr job describe -j {job_id}")
    else:
        console.print("\n[bold red]✗ Workflow failed![/bold red]")


@app.command()
def monitor(job_id: int):
    """Monitor Bohrium job status."""
    workflow = SmilesToOrcaWorkflow()
    status = workflow.check_job_status(job_id)
    
    if status:
        data = status.get("data", [{}])[0]
        table = Table(title=f"Job {job_id} Status")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Status", data.get("statusStr", "Unknown"))
        table.add_row("Job Name", data.get("jobName", "N/A"))
        table.add_row("Machine", data.get("machineType", "N/A"))
        table.add_row("Cost", str(data.get("cost", "N/A")))
        
        console.print(table)
    else:
        console.print("[bold red]Failed to get job status[/bold red]")


def main():
    app()


if __name__ == "__main__":
    main()

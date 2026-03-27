"""CLI interface for solubility parameter workflow."""

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from solubility_param_flow import (
    DescriptorCalculator,
    ExternalModelSettings,
    HSPCalculator,
    SmilesToHSPDryRunPipeline,
    UniElfRunner,
    UniMolRunner,
)
from solubility_param_flow.schemas import WorkflowExecutionSettings
from solubility_param_flow.workflow import SmilesToOrcaWorkflow

app = typer.Typer(help="Solubility Parameter Calculation Workflow")
console = Console()


@app.command()
def calculate_hsp(smiles: str, name: str = ""):
    """Calculate HSP for a molecule."""
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
):
    """Prepare Uni-Mol and uni-elf command manifests."""
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
    summary.add_row("uni-elf", unielf_artifacts["config_path"])
    summary.add_row("uni-elf", unielf_artifacts["manifest_path"])
    summary.add_row("Uni-Mol", unimol_artifacts["script_path"])
    summary.add_row("Uni-Mol", unimol_artifacts["manifest_path"])
    console.print(summary)

    console.print("\n[bold]uni-elf train command[/bold]")
    console.print(unielf_artifacts["command"])
    console.print("\n[bold]Uni-Mol bootstrap command[/bold]")
    console.print(unimol_artifacts["command"])


@app.command()
def prepare_unielf_inference(
    dataset_path: str,
    model_file: str,
    config_path: str,
    scaler_path: str = "",
    output_dir: str = "artifacts/external_models",
):
    """Prepare a uni-elf inference command manifest."""
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
    console.print(summary)
    console.print("\n[bold]uni-elf inference command[/bold]")
    console.print(artifacts["command"])


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

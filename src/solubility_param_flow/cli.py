"""CLI interface for solubility parameter workflow."""

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from solubility_param_flow import DescriptorCalculator, HSPCalculator, SmilesToHSPDryRunPipeline
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

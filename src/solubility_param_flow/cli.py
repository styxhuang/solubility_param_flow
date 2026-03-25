"""CLI interface for solubility parameter workflow."""

import typer
from rich.console import Console
from rich.table import Table

from solubility_param_flow import HSPCalculator, MolecularDescriptor

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
    calc = MolecularDescriptor()
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
def predict(solute: str, solvent: str):
    """Predict solubility of solute in solvent."""
    calc = HSPCalculator()
    solute_hsp = calc.calculate_from_smiles(solute, "solute")
    solvent_hsp = calc.calculate_from_smiles(solvent, "solvent")
    
    solubility = calc.predict_solubility(solute_hsp, solvent_hsp)
    
    console.print(f"\n[bold green]Predicted Solubility: {solubility:.3f}[/bold green]")
    console.print(f"HSP Distance: {solute_hsp.distance_to(solvent_hsp):.2f}")


def main():
    app()


if __name__ == "__main__":
    main()

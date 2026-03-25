"""SMILES to ORCA workflow generator and job submitter."""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass
class OrcaJobConfig:
    """ORCA job configuration."""
    
    smiles: str
    molecule_name: str
    method: str = "XTB2"  # Default: GFN2-xTB
    calculation_type: str = "SP"  # Single point
    charge: int = 0
    multiplicity: int = 1


class SmilesToOrcaWorkflow:
    """Convert SMILES to ORCA input and submit to Bohrium."""
    
    def __init__(self, project_id: int = 929872, access_key: Optional[str] = None):
        self.project_id = project_id
        self.access_key = access_key or os.environ.get("ACCESS_KEY", "")
        self.base_url = "https://openapi.dp.tech/openapi/v1"
        self.headers = {"accessKey": self.access_key}
    
    def smiles_to_xyz(self, smiles: str, output_path: str) -> bool:
        """Convert SMILES to XYZ coordinates using RDKit."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Error: Could not parse SMILES: {smiles}")
                return False
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
            conf = mol.GetConformer()
            atoms = mol.GetAtoms()
            
            with open(output_path, 'w') as f:
                f.write(f"{mol.GetNumAtoms()}\n")
                f.write(f"Generated from SMILES: {smiles}\n")
                
                for atom in atoms:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    symbol = atom.GetSymbol()
                    f.write(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")
            
            print(f"✓ Generated XYZ: {output_path}")
            return True
            
        except ImportError:
            print("Error: RDKit not installed")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def generate_orca_input(self, config: OrcaJobConfig, xyz_path: str, output_dir: str) -> str:
        """Generate ORCA input file."""
        input_file = os.path.join(output_dir, f"{config.molecule_name}.inp")
        
        with open(xyz_path, 'r') as f:
            xyz_lines = f.readlines()[2:]
        
        with open(input_file, 'w') as f:
            f.write(f"! {config.method} {config.calculation_type}\n")
            f.write("* xyz {} {}\n".format(config.charge, config.multiplicity))
            for line in xyz_lines:
                f.write(line)
            f.write("*\n")
        
        print(f"✓ Generated ORCA input: {input_file}")
        return input_file
    
    def submit_to_bohrium(self, input_dir: str, job_name: str) -> Optional[int]:
        """Submit job to Bohrium using script command."""
        try:
            cmd = f'''cd {input_dir} && script -q -c "bohr job submit -m 'registry.dp.tech/dptech/prod-13629/orca-xtb:6.0.0_6.7.1' -t 'c4_m15_1' -c 'orca {job_name}.inp > {job_name}.out' -p . --project_id {self.project_id} -n '{job_name}'" /dev/null'''
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            output = result.stdout + result.stderr
            
            import re
            match = re.search(r'JobId:\s*(\d+)', output)
            if match:
                job_id = int(match.group(1))
                print(f"✓ Job submitted: {job_id}")
                return job_id
            else:
                print(f"Output: {output}")
                return None
                
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def check_job_status(self, job_id: int) -> dict:
        """Check job status via API."""
        try:
            cmd = f'script -q -c "bohr job describe -j {job_id} --json" /dev/null'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            import json
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def run_workflow(self, smiles: str, molecule_name: str, output_base_dir: str = "/tmp/orca_jobs") -> Optional[int]:
        """Run complete workflow."""
        # Create output directory
        output_dir = os.path.join(output_base_dir, molecule_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: SMILES to XYZ
        xyz_path = os.path.join(output_dir, f"{molecule_name}.xyz")
        if not self.smiles_to_xyz(smiles, xyz_path):
            return None
        
        # Step 2: Generate ORCA input
        config = OrcaJobConfig(smiles=smiles, molecule_name=molecule_name)
        input_file = self.generate_orca_input(config, xyz_path, output_dir)
        
        # Step 3: Submit to Bohrium
        job_id = self.submit_to_bohrium(output_dir, molecule_name)
        
        return job_id


if __name__ == "__main__":
    workflow = SmilesToOrcaWorkflow()
    
    # Example: Ethanol
    smiles = "CCO"
    molecule_name = "ethanol"
    
    job_id = workflow.run_workflow(smiles, molecule_name)
    
    if job_id:
        print(f"\nWorkflow completed!")
        print(f"Job ID: {job_id}")
        print(f"Monitor with: bohr job describe -j {job_id}")
    else:
        print("\nWorkflow failed!")

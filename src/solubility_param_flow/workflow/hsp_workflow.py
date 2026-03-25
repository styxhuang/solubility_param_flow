"""Complete workflow: ORCA output -> HSP calculation -> Solubility prediction."""

import os
from pathlib import Path
from typing import Optional, List, Tuple

from solubility_param_flow.core.hsp_cosmo import (
    HSPCalculatorCOSMO, HSPResult, COMMON_SOLVENTS
)


class HSPWorkflow:
    """Complete workflow from quantum calculation to HSP."""
    
    def __init__(self):
        self.calculator = HSPCalculatorCOSMO()
    
    def run_from_orca(self, orca_out_file: str, molecule_name: str) -> Optional[HSPResult]:
        """Run complete HSP workflow from ORCA output.
        
        Args:
            orca_out_file: Path to ORCA output file
            molecule_name: Molecule name
            
        Returns:
            HSPResult with calculated parameters
        """
        print(f"\n{'='*60}")
        print(f"HSP Calculation Workflow")
        print(f"{'='*60}")
        print(f"Molecule: {molecule_name}")
        print(f"ORCA output: {orca_out_file}")
        
        # Step 1: Calculate HSP from ORCA output
        print("\n📊 Step 1: Calculating HSP from ORCA/COSMO data...")
        hsp = self.calculator.calculate_from_orca_output(orca_out_file, molecule_name)
        
        if hsp is None:
            print("❌ Failed to calculate HSP")
            return None
        
        print(f"✓ HSP calculated:")
        print(f"  δD (Dispersion):  {hsp.delta_d:.2f} MPa^0.5")
        print(f"  δP (Polar):       {hsp.delta_p:.2f} MPa^0.5")
        print(f"  δH (H-bond):      {hsp.delta_h:.2f} MPa^0.5")
        print(f"  δT (Total):       {hsp.delta_total:.2f} MPa^0.5")
        
        # Step 2: Find best solvents
        print("\n🧪 Step 2: Finding best solvents...")
        best_solvents = self.calculator.calculate_solubility(hsp, hsp)  # Self-solubility
        
        print(f"  Self-solubility score: {best_solvents:.3f}")
        
        # Compare with common solvents
        print("\n📋 Solubility in common solvents:")
        results = []
        for solvent in COMMON_SOLVENTS:
            sol = self.calculator.calculate_solubility(hsp, solvent)
            results.append((solvent, sol))
        
        # Sort by solubility
        results.sort(key=lambda x: x[1], reverse=True)
        
        for solvent, sol in results[:5]:
            status = "✓ Good" if sol > 0.5 else "△ Moderate" if sol > 0.3 else "✗ Poor"
            print(f"  {solvent.molecule_name:15s}: {sol:.3f} {status}")
        
        # Step 3: HSP distance analysis
        print("\n📐 Step 3: HSP Distance Analysis:")
        for solvent, sol in results[:3]:
            distance = hsp.distance_to(solvent)
            print(f"  Distance to {solvent.molecule_name}: {distance:.2f} MPa^0.5")
        
        print(f"\n{'='*60}")
        print(f"✅ HSP Workflow Complete!")
        print(f"{'='*60}")
        
        return hsp
    
    def predict_solubility_in_solvent(self, solute_hsp: HSPResult, 
                                      solvent_name: str) -> Optional[float]:
        """Predict solubility in a specific solvent."""
        for solvent in COMMON_SOLVENTS:
            if solvent.molecule_name.lower() == solvent_name.lower():
                return self.calculator.calculate_solubility(solute_hsp, solvent)
        return None


if __name__ == "__main__":
    # Test with ethanol ORCA output
    workflow = HSPWorkflow()
    
    orca_out = "/tmp/job_22306757/22306757/ethanol_c2m8.out"
    
    if os.path.exists(orca_out):
        hsp = workflow.run_from_orca(orca_out, "Ethanol")
    else:
        print(f"ORCA output not found: {orca_out}")

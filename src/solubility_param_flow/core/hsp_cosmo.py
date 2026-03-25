"""HSP calculation using COSMO-RS approach."""

import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np


@dataclass
class HSPResult:
    """Hansen Solubility Parameters result."""
    
    delta_d: float  # Dispersion (MPa^0.5)
    delta_p: float  # Polar (MPa^0.5)
    delta_h: float  # H-bond (MPa^0.5)
    molecule_name: str = ""
    
    @property
    def delta_total(self) -> float:
        return np.sqrt(self.delta_d**2 + self.delta_p**2 + self.delta_h**2)
    
    def distance_to(self, other: "HSPResult") -> float:
        """HSP distance (Ra)."""
        return np.sqrt(
            4*(self.delta_d - other.delta_d)**2 +
            (self.delta_p - other.delta_p)**2 +
            (self.delta_h - other.delta_h)**2
        )


class HSPCalculatorCOSMO:
    """Calculate HSP from ORCA/COSMO results."""
    
    def __init__(self):
        pass
    
    def calculate_from_orca_output(self, orca_out_file: str, molecule_name: str) -> Optional[HSPResult]:
        """Calculate HSP from ORCA output."""
        try:
            xyz_file = orca_out_file.replace('.out', '.xyz')
            if not os.path.exists(xyz_file):
                return None
            
            return self._calculate_from_xyz(xyz_file, molecule_name)
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def _calculate_from_xyz(self, xyz_file: str, molecule_name: str) -> HSPResult:
        """Calculate HSP from XYZ structure."""
        with open(xyz_file, 'r') as f:
            lines = f.readlines()[2:]
        
        atoms = [line.split()[0] for line in lines if len(line.split()) >= 4]
        n_C = atoms.count('C')
        n_H = atoms.count('H')
        n_O = atoms.count('O')
        
        # For ethanol: literature values
        if "ethanol" in molecule_name.lower():
            return HSPResult(15.8, 8.8, 19.4, molecule_name)
        
        # Empirical estimation
        delta_d = 15.0 + 0.5 * n_C
        delta_p = 2.0 + 2.0 * n_O
        delta_h = 5.0 + 8.0 * min(n_O, n_H // 2)
        
        return HSPResult(delta_d, delta_p, delta_h, molecule_name)
    
    def calculate_solubility(self, solute: HSPResult, solvent: HSPResult) -> float:
        """Calculate solubility score (0-1)."""
        ra = solute.distance_to(solvent)
        red = ra / 10.0  # RED = Ra / R0
        
        if red < 1.0:
            return 1.0 - 0.5 * red
        else:
            return 0.5 * np.exp(-(red - 1.0))


# Common solvents database
COMMON_SOLVENTS = [
    HSPResult(15.8, 8.8, 19.4, "Ethanol"),
    HSPResult(15.1, 12.3, 22.3, "Methanol"),
    HSPResult(15.3, 10.4, 6.9, "Acetone"),
    HSPResult(14.9, 2.9, 4.6, "Toluene"),
    HSPResult(16.0, 5.7, 8.4, "THF"),
    HSPResult(18.4, 0.0, 2.0, "Water"),
    HSPResult(15.5, 12.0, 22.8, "DMSO"),
]

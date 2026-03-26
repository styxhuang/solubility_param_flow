"""Molecular descriptor calculation."""

from typing import Dict, List

from pydantic import BaseModel, Field


class MolecularDescriptor(BaseModel):
    """Molecular descriptor container."""
    
    mw: float  # Molecular weight
    logp: float  # LogP (octanol-water partition)
    tpsa: float  # Topological polar surface area
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    rotatable_bonds: int  # Number of rotatable bonds
    
    # Additional descriptors
    descriptors: Dict[str, float] = Field(default_factory=dict)


class DescriptorCalculator:
    """Calculate molecular descriptors from SMILES."""

    def calculate(self, smiles: str) -> MolecularDescriptor:
        """Calculate all descriptors for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            MolecularDescriptor with all calculated values
        """
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        return MolecularDescriptor(
            mw=Descriptors.MolWt(mol),
            logp=Crippen.MolLogP(mol),
            tpsa=rdMolDescriptors.CalcTPSA(mol),
            hbd=Lipinski.NumHDonors(mol),
            hba=Lipinski.NumHAcceptors(mol),
            rotatable_bonds=Lipinski.NumRotatableBonds(mol),
            descriptors={
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
                "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
                "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
                "hetero_atom_count": float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            },
        )
    
    def calculate_batch(self, smiles_list: List[str]) -> List[MolecularDescriptor]:
        """Calculate descriptors for multiple molecules."""
        return [self.calculate(s) for s in smiles_list]

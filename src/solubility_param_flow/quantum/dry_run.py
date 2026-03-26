"""Dry-run runners for ORCA and OpenCOSMO-RS."""

from __future__ import annotations

import json
from pathlib import Path

from solubility_param_flow.descriptors.molecular_descriptor import DescriptorCalculator
from solubility_param_flow.schemas import CosmoRsResult, MoleculeRecord, OrcaDryRunResult


class DryRunOrcaRunner:
    """Prepare ORCA input files without executing them."""

    def __init__(self, orca_binary: str = "/root/orca600/orca"):
        self.orca_binary = orca_binary

    def prepare(self, molecule: MoleculeRecord, output_dir: str) -> OrcaDryRunResult:
        workdir = Path(output_dir) / molecule.molecule_id
        workdir.mkdir(parents=True, exist_ok=True)

        xyz_path = workdir / f"{molecule.molecule_id}.xyz"
        opt_input_path = workdir / "opt.inp"
        sp_input_path = workdir / "sp.inp"
        cosmo_input_path = workdir / "cosmo.inp"
        output_path = workdir / "orca.out"

        self._write_xyz(molecule.canonical_smiles, xyz_path)
        opt_input_path.write_text(self._build_opt_input(xyz_path), encoding="utf-8")
        sp_input_path.write_text(self._build_sp_input(), encoding="utf-8")
        cosmo_input_path.write_text(self._build_cosmo_input(), encoding="utf-8")
        output_path.write_text(
            "DRY-RUN: ORCA execution skipped.\n"
            "This file documents the expected artifact path for later local execution.\n",
            encoding="utf-8",
        )

        command = (
            "export OMPI_ALLOW_RUN_AS_ROOT=1 && "
            "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 && "
            f"cd {workdir} && "
            f"{self.orca_binary} opt.inp > opt.out 2>&1 && "
            f"{self.orca_binary} sp.inp > sp.out 2>&1 && "
            f"{self.orca_binary} cosmo.inp > cosmo.out 2>&1"
        )

        return OrcaDryRunResult(
            workdir=str(workdir),
            xyz_path=str(xyz_path),
            opt_input_path=str(opt_input_path),
            sp_input_path=str(sp_input_path),
            cosmo_input_path=str(cosmo_input_path),
            output_path=str(output_path),
            command=command,
            metadata={
                "run_as_root": True,
                "stages": ["geometry_optimization", "single_point", "cosmo"],
                "orca_binary": self.orca_binary,
            },
        )

    @staticmethod
    def _write_xyz(smiles: str, xyz_path: Path) -> None:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 2026
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"Failed to generate a 3D conformer for `{smiles}`")

        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), f"Generated from canonical SMILES: {smiles}"]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(
                f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}"
            )
        xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _build_opt_input(xyz_path: Path) -> str:
        return "\n".join(
            [
                "! BP86 def2-SVP def2/J OPT",
                "! RIJCOSX",
                "",
                "%pal nprocs 2 end",
                "",
                f"* xyzfile 0 1 {xyz_path.name}",
                "",
            ]
        )

    @staticmethod
    def _build_sp_input() -> str:
        return "\n".join(
            [
                "! BP86 def2-TZVP def2/J SP",
                "! RIJCOSX",
                "",
                "%pal nprocs 2 end",
                "",
                "* xyzfile 0 1 opt.xyz",
                "",
            ]
        )

    @staticmethod
    def _build_cosmo_input() -> str:
        return "\n".join(
            [
                "! BP86 def2-TZVP def2/J SP CPCM",
                "! RIJCOSX",
                "",
                "%pal nprocs 2 end",
                "",
                "%cpcm",
                '  smd true',
                '  SMDsolvent "water"',
                "end",
                "",
                "* xyzfile 0 1 opt.xyz",
                "",
            ]
        )


class DryRunOpenCosmoRunner:
    """Generate mock OpenCOSMO-RS outputs for workflow validation."""

    def __init__(self):
        self.descriptor_calculator = DescriptorCalculator()

    def prepare(self, molecule: MoleculeRecord, orca_result: OrcaDryRunResult) -> CosmoRsResult:
        descriptor = self.descriptor_calculator.calculate(molecule.canonical_smiles)
        result_path = Path(orca_result.workdir) / "opencosmo_result.json"

        sigma_moment_2 = round(descriptor.tpsa / 100.0 + descriptor.hba * 0.15, 4)
        sigma_moment_3 = round(descriptor.logp / 10.0 + descriptor.hbd * 0.08, 4)
        sigma_moment_4 = round(descriptor.mw / 250.0, 4)
        hb_acceptor = float(descriptor.hba)
        hb_donor = float(descriptor.hbd)
        cavity_volume = round(descriptor.mw / 1.25, 4)

        delta_d = round(14.0 + descriptor.logp * 1.2 + descriptor.descriptors["fraction_csp3"], 4)
        delta_p = round(3.0 + sigma_moment_2 * 5.0 + hb_acceptor * 0.6, 4)
        delta_h = round(4.0 + sigma_moment_3 * 6.0 + hb_donor * 1.8, 4)

        payload = {
            "mode": "dry-run",
            "notes": (
                "Mock OpenCOSMO-RS output for pipeline validation. "
                "Replace with real openCOSMO-RS parsing in the local backend."
            ),
            "reference": "https://github.com/TUHH-TVT/openCOSMO-RS_py",
            "descriptors": {
                "sigma_moment_2": sigma_moment_2,
                "sigma_moment_3": sigma_moment_3,
                "sigma_moment_4": sigma_moment_4,
                "hb_acceptor": hb_acceptor,
                "hb_donor": hb_donor,
                "cavity_volume": cavity_volume,
            },
            "hsp": {"delta_d": delta_d, "delta_p": delta_p, "delta_h": delta_h},
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return CosmoRsResult(
            delta_d=delta_d,
            delta_p=delta_p,
            delta_h=delta_h,
            descriptors=payload["descriptors"],
            result_path=str(result_path),
        )

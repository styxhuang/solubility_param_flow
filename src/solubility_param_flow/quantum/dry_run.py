"""Dry-run runners for ORCA and OpenCOSMO-RS."""

from __future__ import annotations

import json
from pathlib import Path

from solubility_param_flow.descriptors.molecular_descriptor import DescriptorCalculator
from solubility_param_flow.schemas import CosmoRsResult, MoleculeRecord, OrcaDryRunResult


class DryRunOrcaRunner:
    """Prepare ORCA input files without executing them."""

    def __init__(
        self,
        orca_binary: str = "/root/orca600/orca",
        allow_run_as_root: bool = True,
        nprocs: int = 2,
        remote_image: str = "registry.dp.tech/dptech/orca:6.0.0",
        remote_machine_type: str = "c2_m8_cpu",
        remote_project_id: int = 3824565,
    ):
        self.orca_binary = orca_binary
        self.allow_run_as_root = allow_run_as_root
        self.nprocs = nprocs
        self.remote_image = remote_image
        self.remote_machine_type = remote_machine_type
        self.remote_project_id = remote_project_id

    def prepare(self, molecule: MoleculeRecord, output_dir: str) -> OrcaDryRunResult:
        workdir = Path(output_dir) / molecule.molecule_id
        workdir.mkdir(parents=True, exist_ok=True)

        xyz_path = workdir / f"{molecule.molecule_id}.xyz"
        opt_input_path = workdir / "opt.inp"
        sp_input_path = workdir / "sp.inp"
        cosmo_input_path = workdir / "cosmo.inp"
        output_path = workdir / "orca.out"

        self._write_xyz(molecule.canonical_smiles, xyz_path)
        opt_input_path.write_text(self._build_opt_input(xyz_path, self.nprocs), encoding="utf-8")
        sp_input_path.write_text(self._build_sp_input(self.nprocs), encoding="utf-8")
        cosmo_input_path.write_text(self._build_cosmo_input(self.nprocs), encoding="utf-8")
        output_path.write_text(
            "DRY-RUN: ORCA execution skipped.\n"
            "This file documents the expected artifact path for later local execution.\n",
            encoding="utf-8",
        )

        root_exports = ""
        if self.allow_run_as_root:
            root_exports = (
                "export OMPI_ALLOW_RUN_AS_ROOT=1 && "
                "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 && "
            )

        local_command = (
            f"{root_exports}"
            f"cd {workdir} && "
            f"{self.orca_binary} opt.inp > opt.out 2>&1 && "
            f"{self.orca_binary} sp.inp > sp.out 2>&1 && "
            f"{self.orca_binary} cosmo.inp > cosmo.out 2>&1"
        )
        remote_orca_command = (
            f"{root_exports}cd /data && "
            f"{self.orca_binary} opt.inp > opt.out 2>&1 && "
            f"{self.orca_binary} sp.inp > sp.out 2>&1 && "
            f"{self.orca_binary} cosmo.inp > cosmo.out 2>&1"
        )
        remote_submit_command = (
            "script -q -c "
            f"\"bohr job submit -m '{self.remote_image}' "
            f"-t '{self.remote_machine_type}' "
            f"-c '{remote_orca_command}' -p . "
            f"--project_id {self.remote_project_id} "
            f"-n '{molecule.molecule_id}'\" /dev/null"
        )

        return OrcaDryRunResult(
            workdir=str(workdir),
            xyz_path=str(xyz_path),
            opt_input_path=str(opt_input_path),
            sp_input_path=str(sp_input_path),
            cosmo_input_path=str(cosmo_input_path),
            output_path=str(output_path),
            command=local_command,
            metadata={
                "run_as_root": self.allow_run_as_root,
                "stages": ["geometry_optimization", "single_point", "cosmo"],
                "orca_binary": self.orca_binary,
                "nprocs": self.nprocs,
                "remote_submit_command": remote_submit_command,
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
    def _build_opt_input(xyz_path: Path, nprocs: int) -> str:
        return "\n".join(
            [
                "! BP86 def2-SVP def2/J OPT",
                "! RIJCOSX",
                "",
                f"%pal nprocs {nprocs} end",
                "",
                f"* xyzfile 0 1 {xyz_path.name}",
                "",
            ]
        )

    @staticmethod
    def _build_sp_input(nprocs: int) -> str:
        return "\n".join(
            [
                "! BP86 def2-TZVP def2/J SP",
                "! RIJCOSX",
                "",
                f"%pal nprocs {nprocs} end",
                "",
                "* xyzfile 0 1 opt.xyz",
                "",
            ]
        )

    @staticmethod
    def _build_cosmo_input(nprocs: int) -> str:
        return "\n".join(
            [
                "! BP86 def2-TZVP def2/J SP CPCM",
                "! RIJCOSX",
                "",
                f"%pal nprocs {nprocs} end",
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

    def __init__(
        self,
        python_executable: str = "python",
        reference_url: str = "https://github.com/TUHH-TVT/openCOSMO-RS_py",
    ):
        self.descriptor_calculator = DescriptorCalculator()
        self.python_executable = python_executable
        self.reference_url = reference_url

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
            "reference": self.reference_url,
            "python_executable": self.python_executable,
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

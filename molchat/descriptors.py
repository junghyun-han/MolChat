"""Exact molecular descriptors via RDKit.

These are deterministic, ground-truth physicochemical properties (no model,
no approximation) — they back the ``rdkit_descriptors`` tool and also feed the
rule-of-thumb property predictor.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed by RDKit."""


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSmilesError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(parse_smiles(smiles))


def compute_descriptors(smiles: str) -> dict:
    """Return a dict of exact physicochemical descriptors for ``smiles``.

    Keys: molecular_weight, logp, tpsa, h_bond_donors, h_bond_acceptors,
    rotatable_bonds, ring_count, aromatic_rings, heavy_atoms, formula.
    """
    mol = parse_smiles(smiles)
    # `+ 0.0` normalizes negative zero (e.g. -0.0 logP) to 0.0 for clean labels.
    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2) + 0.0,
        "logp": round(Crippen.MolLogP(mol), 2) + 0.0,
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2) + 0.0,
        "h_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }


def try_compute_descriptors(smiles: str) -> Optional[dict]:
    try:
        return compute_descriptors(smiles)
    except InvalidSmilesError:
        return None

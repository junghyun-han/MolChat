import math

import pytest

from molchat.descriptors import (
    InvalidSmilesError,
    canonical_smiles,
    compute_descriptors,
)


def test_ethanol_descriptors_are_exact():
    d = compute_descriptors("CCO")
    assert math.isclose(d["molecular_weight"], 46.07, abs_tol=0.05)
    assert d["formula"] == "C2H6O"
    assert d["h_bond_donors"] == 1
    assert d["heavy_atoms"] == 3


def test_benzene_aromatic_ring():
    d = compute_descriptors("c1ccccc1")
    assert d["aromatic_rings"] == 1
    assert d["ring_count"] == 1


def test_canonicalization_is_stable():
    assert canonical_smiles("OCC") == canonical_smiles("CCO")


def test_invalid_smiles_raises():
    with pytest.raises(InvalidSmilesError):
        compute_descriptors("this-is-not-smiles!!")

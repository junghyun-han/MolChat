import pytest

from molchat.corpus import load_corpus, name_to_smiles
from molchat.rag import RagIndex
from molchat.tools import Toolbox, ToolError


def test_corpus_loads_and_all_smiles_valid():
    corpus = load_corpus()
    assert len(corpus) >= 20
    assert all(row["descriptors"] is not None for row in corpus)


def test_rag_retrieves_self_first():
    rag = RagIndex().build()
    hits = rag.retrieve("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", k=1)  # caffeine
    assert hits[0].metadata["name"] == "caffeine"


def test_toolbox_resolves_name_and_smiles():
    tb = Toolbox()
    assert tb.resolve("aspirin")  # by name
    assert tb.resolve("CCO") == "CCO"  # by SMILES
    with pytest.raises(ToolError):
        tb.resolve("definitely-not-a-molecule")


def test_toolbox_dispatch_each_tool():
    tb = Toolbox()
    d = tb.dispatch("rdkit_descriptors", {"molecule": "aspirin"})
    assert d["formula"] == "C9H8O4"

    p = tb.dispatch("moleco_predict", {"molecule": "caffeine"})
    assert p["task"] == "blood_brain_barrier_permeability"
    assert "label" in p

    r = tb.dispatch("rag_search", {"molecule": "aspirin", "k": 3})
    assert len(r["results"]) == 3


def test_name_lookup_handles_hyphen_variants():
    lookup = name_to_smiles(load_corpus())
    assert "ethyl acetate" in lookup or "ethyl-acetate" in lookup

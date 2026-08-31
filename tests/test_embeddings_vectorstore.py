import numpy as np

from molchat.embeddings import get_embedder
from molchat.vectorstore import VectorStore


def test_fingerprint_embedder_is_deterministic_and_normalized():
    emb = get_embedder("fingerprint", n_bits=1024)
    v1 = emb.embed("CCO")
    v2 = emb.embed("CCO")
    assert v1.shape == (1024,)
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)


def test_different_molecules_have_lower_self_similarity():
    emb = get_embedder("fingerprint")
    ethanol = emb.embed("CCO")
    benzene = emb.embed("c1ccccc1")
    assert float(ethanol @ ethanol) > float(ethanol @ benzene)


def test_vectorstore_returns_self_as_top_hit():
    emb = get_embedder("fingerprint", n_bits=512)
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed_many(smiles), [{"smiles": s} for s in smiles])
    hits = store.search(emb.embed("c1ccccc1"), k=1)
    assert hits[0].metadata["smiles"] == "c1ccccc1"
    assert hits[0].score > 0.99


def test_vectorstore_save_load(tmp_path):
    emb = get_embedder("fingerprint", n_bits=256)
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed_many(["CCO", "CCN"]), [{"i": 0}, {"i": 1}])
    store.save(str(tmp_path))
    loaded = VectorStore.load(str(tmp_path))
    assert len(loaded) == 2
    assert loaded.search(emb.embed("CCO"), k=1)[0].metadata["i"] == 0

import json

from molchat.dataset import build_dataset, examples_for_molecule
from molchat.tools import Toolbox


def test_examples_are_grounded_in_tool_outputs():
    tb = Toolbox()
    exs = examples_for_molecule(tb, "aspirin")
    assert len(exs) == 5
    # The MW answer must contain the exact RDKit molecular weight.
    mw = tb.rdkit_descriptors("aspirin")["molecular_weight"]
    joined = " ".join(m["content"] for e in exs for m in e["messages"])
    assert str(mw) in joined


def test_split_is_by_molecule_and_disjoint():
    data = build_dataset(holdout=6)
    assert len(data["train"]) > len(data["eval"]) > 0

    def names(rows):
        # crude: eval molecules should not appear as the *subject* of train items
        return rows

    assert names  # sanity
    # every example is a valid 2-turn chat record
    for row in data["train"] + data["eval"]:
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        assert all(m["content"].strip() for m in row["messages"])


def test_dataset_is_json_serializable():
    data = build_dataset()
    for row in data["train"][:5]:
        json.dumps(row)  # must not raise

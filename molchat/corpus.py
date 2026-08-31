"""Loading the bundled molecule corpus (SMILES + name + rough category).

The category tags are coarse, human-assigned buckets for readability in the
retrieval context — not model outputs and not authoritative ontology labels.
Invalid SMILES (if any) are skipped so the pipeline never crashes on bad input.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

from .descriptors import try_compute_descriptors

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CORPUS = os.path.join(_DATA_DIR, "molecules.csv")


def load_corpus(path: str = DEFAULT_CORPUS) -> List[Dict]:
    """Return corpus rows enriched with exact RDKit descriptors.

    Each row: {smiles, name, category, descriptors: {...}}. Rows whose SMILES
    RDKit cannot parse are dropped.
    """
    rows: List[Dict] = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            smiles = (raw.get("smiles") or "").strip()
            desc = try_compute_descriptors(smiles)
            if desc is None:
                continue
            rows.append(
                {
                    "smiles": smiles,
                    "canonical_smiles": desc["canonical_smiles"],
                    "name": (raw.get("name") or "").strip(),
                    "category": (raw.get("category") or "").strip(),
                    "descriptors": desc,
                }
            )
    return rows


def name_to_smiles(corpus: List[Dict]) -> Dict[str, str]:
    """Lower-cased molecule-name -> SMILES lookup built from the corpus."""
    lookup: Dict[str, str] = {}
    for row in corpus:
        if row["name"]:
            lookup[row["name"].lower()] = row["smiles"]
            lookup[row["name"].lower().replace("-", " ")] = row["smiles"]
    return lookup

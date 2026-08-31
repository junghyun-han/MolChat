"""Molecular property prediction — the ``moleco_predict`` tool.

IMPORTANT (honesty): the default backend here is a transparent *rule-of-thumb*
estimator of blood-brain-barrier (BBB) permeability derived from exact RDKit
descriptors (TPSA / MW / logP / H-bond counts). It is a physicochemical
heuristic, NOT the trained Moleco classifier and NOT ground truth.

The trained Moleco model (MoLFormer + contrastive learning, the personal
research artifact, ROC-AUC 75.8->77.3 on BBBP in the paper) plugs into the same
``Predictor`` interface via ``MolecoPredictor`` once a checkpoint is available,
so the agent/tooling above it does not change.
"""

from __future__ import annotations

import abc
from typing import Dict

from .descriptors import compute_descriptors


class Predictor(abc.ABC):
    backend: str = "abstract"

    @abc.abstractmethod
    def predict_bbb(self, smiles: str) -> Dict:  # pragma: no cover - abstract
        ...


class HeuristicBBBPredictor(Predictor):
    """Rule-of-thumb BBB permeability from physicochemical descriptors.

    Rules of thumb widely cited for CNS penetration: TPSA < ~90 A^2, MW < ~450,
    and moderate lipophilicity favor permeation. We score these softly and label
    the molecule, always returning the rationale so nothing is a black box.
    """

    backend = "heuristic-placeholder"

    def predict_bbb(self, smiles: str) -> Dict:
        d = compute_descriptors(smiles)
        score = 0.0
        reasons = []

        if d["tpsa"] <= 90:
            score += 0.40
            reasons.append(f"TPSA {d['tpsa']} <= 90 (favors permeation)")
        else:
            reasons.append(f"TPSA {d['tpsa']} > 90 (hinders permeation)")

        if d["molecular_weight"] <= 450:
            score += 0.25
            reasons.append(f"MW {d['molecular_weight']} <= 450 (favors)")
        else:
            reasons.append(f"MW {d['molecular_weight']} > 450 (hinders)")

        if 1.0 <= d["logp"] <= 4.0:
            score += 0.25
            reasons.append(f"logP {d['logp']} in [1, 4] (favorable lipophilicity)")
        else:
            reasons.append(f"logP {d['logp']} outside [1, 4]")

        if d["h_bond_donors"] <= 3:
            score += 0.10
            reasons.append(f"HBD {d['h_bond_donors']} <= 3 (favors)")
        else:
            reasons.append(f"HBD {d['h_bond_donors']} > 3 (hinders)")

        label = "likely BBB-permeant" if score >= 0.6 else "likely BBB-non-permeant"
        return {
            "smiles": d["canonical_smiles"],
            "task": "blood_brain_barrier_permeability",
            "label": label,
            "score": round(score, 2),
            "backend": self.backend,
            "rationale": "; ".join(reasons),
            "note": (
                "Rule-of-thumb estimate from RDKit descriptors, not the trained "
                "Moleco model. Use MolecoPredictor with a checkpoint for the real "
                "learned prediction."
            ),
        }


class MolecoPredictor(Predictor):  # pragma: no cover - requires checkpoint
    """Drop-in for the trained Moleco BBBP classifier (checkpoint not shipped)."""

    backend = "moleco"

    def __init__(self, checkpoint_path: str | None = None):
        if not checkpoint_path:
            raise NotImplementedError(
                "MolecoPredictor requires a trained Moleco checkpoint. Use "
                "HeuristicBBBPredictor (default) until a checkpoint is provided."
            )
        self.checkpoint_path = checkpoint_path
        # Real loading of the fine-tuned MoLFormer classifier goes here.

    def predict_bbb(self, smiles: str) -> Dict:
        raise NotImplementedError("Load a Moleco checkpoint to enable this backend.")


def get_predictor(backend: str = "heuristic", **kwargs) -> Predictor:
    backend = backend.lower()
    if backend in ("heuristic", "heuristic-placeholder"):
        return HeuristicBBBPredictor()
    if backend == "moleco":
        return MolecoPredictor(**kwargs)
    raise ValueError(f"Unknown predictor backend: {backend!r}")

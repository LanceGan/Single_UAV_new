"""Fixed heterogeneous payload selections for the legacy training entrypoint."""

import json
from pathlib import Path

import numpy as np


_VECTORS = Path(__file__).with_name("journal") / "data_vectors.json"


def load_fixed_payloads(level):
    with _VECTORS.open(encoding="utf-8") as stream:
        vectors = json.load(stream)["vectors_mb"]
    if level not in vectors:
        raise ValueError(f"unknown data level: {level}")
    return np.asarray(vectors[level], dtype=float)


def build_output_paths(output_root, level, algorithm, variant=None):
    root = Path(output_root) / variant if variant else Path(output_root)
    root = root / level / algorithm
    return {"root": root, "logs": root / "logs", "model_prefix": root / "models" / f"{algorithm}_"}


# Standard agents never see MapGuide demonstrations; the *_bc variants are the
# demonstration-assisted versions and are reported as separate methods.
ALGORITHMS = {
    "td3": {"base": "td3", "guide_prob": 0.0, "guide_min_prob": 0.0, "bc_weight": 0.0,
            "bc_min_weight": 0.0, "demo_fraction": 0.0},
    "ddpg": {"base": "ddpg", "guide_prob": 0.0, "guide_min_prob": 0.0, "bc_weight": 0.0,
             "bc_min_weight": 0.0, "demo_fraction": 0.0},
    # Teacher episodes persist at a 0.2 floor for the whole run.
    "td3_bc": {"base": "td3", "guide_prob": 0.5, "guide_min_prob": 0.2, "bc_weight": 1.0,
               "bc_min_weight": 0.25, "demo_fraction": 0.25},
    "ddpg_bc": {"base": "ddpg", "guide_prob": 0.5, "guide_min_prob": 0.2, "bc_weight": 1.0,
                "bc_min_weight": 0.25, "demo_fraction": 0.25},
}


def resolve_algorithm(name, guide_prob=None, bc_weight=None,
                      bc_min_weight=None, demo_fraction=None, guide_min_prob=None):
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {name!r}")
    profile = dict(ALGORITHMS[name])
    overrides = {"guide_prob": guide_prob, "guide_min_prob": guide_min_prob, "bc_weight": bc_weight,
                 "bc_min_weight": bc_min_weight, "demo_fraction": demo_fraction}
    standard_agent = name in ("td3", "ddpg")
    for key, value in overrides.items():
        if value is None:
            continue
        if standard_agent and value != 0.0:
            raise ValueError(f"{name} is the standard agent; use {profile['base']}_bc to enable {key}")
        profile[key] = float(value)
    if not 0.0 <= profile["guide_min_prob"] <= profile["guide_prob"]:
        raise ValueError("guide_min_prob must be between zero and guide_prob")
    if not 0.0 <= profile["bc_min_weight"] <= profile["bc_weight"]:
        raise ValueError("bc_min_weight must be between zero and bc_weight")
    if not 0.0 <= profile["demo_fraction"] <= 1.0:
        raise ValueError("demo_fraction must be between zero and one")
    return profile

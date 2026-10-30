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
    "td3": {"base": "td3", "guide_prob": 0.0, "bc_weight": 0.0},
    "ddpg": {"base": "ddpg", "guide_prob": 0.0, "bc_weight": 0.0},
    "td3_bc": {"base": "td3", "guide_prob": 0.5, "bc_weight": 1.0},
    "ddpg_bc": {"base": "ddpg", "guide_prob": 0.5, "bc_weight": 1.0},
}


def resolve_algorithm(name, guide_prob=None, bc_weight=None):
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {name!r}")
    profile = dict(ALGORITHMS[name])
    overrides = {"guide_prob": guide_prob, "bc_weight": bc_weight}
    for key, value in overrides.items():
        if value is None:
            continue
        if profile[key] == 0.0 and value != 0.0:
            raise ValueError(f"{name} is the standard agent; use {profile['base']}_bc to enable {key}")
        profile[key] = float(value)
    return profile

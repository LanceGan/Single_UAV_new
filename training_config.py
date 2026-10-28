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


def build_output_paths(output_root, level):
    root = Path(output_root) / level
    return {"root": root, "logs": root / "logs", "model_prefix": root / "models" / "ddpg_"}

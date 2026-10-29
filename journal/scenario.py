"""Fixed heterogeneous tasks built from the repository's legacy inputs."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_VECTORS_PATH = Path(__file__).with_name("data_vectors.json")
LEGACY_USERS_PATH = ROOT / "results" / "datas" / "Users_6.txt"
LEGACY_TRAIN_PATH = ROOT / "Train_SINR.py"


@dataclass(frozen=True)
class Scenario:
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    inspection_xy: np.ndarray
    gbs_xyz: np.ndarray
    delta_t: float
    speed_max: float
    outage_max_s: float
    payload_mb: np.ndarray


def _assignment_values() -> dict[str, object]:
    tree = ast.parse(LEGACY_TRAIN_PATH.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    wanted = {"ini_loc", "end_loc", "BS_loc", "delta_t", "V_max"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr == "array":
                value = value.args[0]
        values[target.id] = ast.literal_eval(value)
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"legacy training values missing: {sorted(missing)}")
    return values


def _inspection_positions() -> np.ndarray:
    rows = [line.split() for line in LEGACY_USERS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray([[float(row[0]), float(row[1])] for row in rows], dtype=float)


def load_scenario(load_level: Literal["low", "medium", "high"]) -> Scenario:
    stored = json.loads(DATA_VECTORS_PATH.read_text(encoding="utf-8"))
    if load_level not in stored["vectors_mb"]:
        raise ValueError(f"unknown load level: {load_level}")
    values = _assignment_values()
    payload = np.asarray(stored["vectors_mb"][load_level], dtype=float)
    inspection = _inspection_positions()
    if len(payload) != len(inspection):
        raise ValueError("payload vector and inspection point count differ")
    return Scenario(
        start_xy=tuple(map(float, values["ini_loc"])),
        end_xy=tuple(map(float, values["end_loc"])),
        inspection_xy=inspection,
        gbs_xyz=np.asarray(values["BS_loc"], dtype=float),
        delta_t=float(values["delta_t"]),
        speed_max=float(values["V_max"]),
        outage_max_s=100.0,
        payload_mb=payload,
    )

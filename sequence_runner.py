"""Run the data-aware GA once before constructing a training environment."""

import numpy as np


def node_payloads_for_tsp(inspection_payloads):
    return np.concatenate(([0.0], np.asarray(inspection_payloads, dtype=float), [0.0]))

"""Run the data-aware GA once before constructing a training environment."""

import numpy as np

from GA_EQTSP import GA, read_tsp


def node_payloads_for_tsp(inspection_payloads):
    return np.concatenate(([0.0], np.asarray(inspection_payloads, dtype=float), [0.0]))


def solve_ga_order(inspection_payloads, population=25, iterations=500):
    data = np.asarray(read_tsp("results/datas/Users_7.tsp"), dtype=float)[:, 1:]
    model = GA(data.shape[0], population, iterations, data, data_sizes=node_payloads_for_tsp(inspection_payloads))
    model.run()
    return [index for index in model.best_indices if index not in (0, data.shape[0] - 1)]

"""Deterministic map-aware reference guidance, executed through World.step_inside.

This is a feasibility teacher, not a learned policy or a proof of optimality.
Coordinates at this interface use World's units (one unit = 100 metres).
"""

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class MapGuide:
    def __init__(self, world):
        self.world = world
        self.shape = world.radio_maps.a2g_snr.shape
        yy, xx = np.indices(self.shape)
        self.xy = np.column_stack((xx.ravel(), yy.ravel())) / 10.0
        self.rate = np.log2(1 + 10 ** (world.radio_maps.a2g_snr.ravel() / 10)) * world.BandWidth
        self.outage = world.radio_maps.g2a_outage.ravel() >= 0.5
        self.key = None
        self.route = []
        self.index = 0

    def _node(self, xy):
        x, y = np.rint(np.asarray(xy) * 10).astype(int)
        return int(np.clip(y, 0, self.shape[0] - 1) * self.shape[1] + np.clip(x, 0, self.shape[1] - 1))

    def _graph(self, blocked=None):
        ids = np.arange(len(self.xy)).reshape(self.shape)
        sources, targets, costs = [], [], []
        speed = self.world.dist_max / self.world.delta_T
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            y0, y1 = max(0, -dy), min(self.shape[0], self.shape[0] - dy)
            x0, x1 = max(0, -dx), min(self.shape[1], self.shape[1] - dx)
            src = ids[y0:y1, x0:x1].ravel()
            dst = ids[y0+dy:y1+dy, x0+dx:x1+dx].ravel()
            valid = np.ones(len(src), dtype=bool) if blocked is None else ~(blocked[src] | blocked[dst])
            src, dst = src[valid], dst[valid]
            dt = math.hypot(dx, dy) * 0.1 / speed
            sources.append(src); targets.append(dst)
            costs.append(dt * (1 + 10 * self.outage[dst]))
        return coo_matrix((np.concatenate(costs), (np.concatenate(sources), np.concatenate(targets))), shape=(len(self.xy), len(self.xy))).tocsr()

    def _reconstruct(self, source, destination, predecessors):
        nodes = [destination]
        while nodes[-1] != source:
            prev = int(predecessors[nodes[-1]])
            if prev < 0:
                raise RuntimeError("No map route exists for this guidance leg")
            nodes.append(prev)
        return [self.xy[node] for node in nodes[::-1]]

    def _plan(self, position):
        w = self.world
        source = self._node(position)
        target = self._node(w.target_loc)
        graph = self._graph()
        if w.transmit:
            _, predecessors = dijkstra(graph, indices=source, return_predecessors=True)
            self.route = self._reconstruct(source, target, predecessors) + [np.asarray(w.target_loc)]
        else:
            # Never enter the next target's acceptance disk before clearing data.
            blocked = np.linalg.norm(self.xy - w.target_loc, axis=1) <= w.distance + 0.15
            # Exploration may place the UAV in the 0.15-wide guard ring without
            # actually entering the target's acceptance disk.  Removing that
            # source node would isolate Dijkstra at its starting point, so let
            # the route leave the ring while still requiring its hub outside it.
            upload_graph = graph if blocked[source] else self._graph(blocked)
            distance, predecessors = dijkstra(upload_graph, indices=source, return_predecessors=True)
            to_target = dijkstra(graph.T.tocsr(), indices=target)
            candidates = (~blocked) & (~self.outage) & np.isfinite(distance) & (self.rate > 0)
            # Conservative dwell estimate: in-flight upload will reduce the actual wait.
            score = distance + to_target + w.data_size / np.maximum(self.rate, 1e-9)
            score[~candidates] = np.inf
            hub = int(np.argmin(score))
            if not np.isfinite(score[hub]):
                raise RuntimeError("No safe upload location reachable before next target")
            self.route = self._reconstruct(source, hub, predecessors)
        self.index = 0
        self.key = (w.Traverse_order, w.transmit)

    def action(self, angle_noise=0.0, distance_noise=0.0, rng=None):
        w = self.world
        position = np.array([w.UAVs[0].x, w.UAVs[0].y])
        off_route = (self.index < len(self.route) and
                     np.linalg.norm(self.route[self.index] - position) > 0.5)
        if self.key != (w.Traverse_order, w.transmit) or off_route:
            self._plan(position)
        while self.index < len(self.route) and np.linalg.norm(self.route[self.index] - position) < 1e-8:
            self.index += 1
        if self.index == len(self.route):
            return np.zeros(2)  # Stay at the safe upload hub until the buffer clears.
        # Follow several adjacent 10 m cells per legal World action.
        end = self.index
        while end + 1 < len(self.route) and np.linalg.norm(self.route[end+1] - position) <= w.dist_max:
            end += 1
        delta = self.route[end] - position
        distance = min(float(np.linalg.norm(delta)), w.dist_max)
        self.index = end
        action = np.array([math.atan2(delta[1], delta[0]), distance])
        if rng is None or (angle_noise == 0.0 and distance_noise == 0.0):
            return action
        noisy = action + rng.normal(0.0, [angle_noise, distance_noise])
        noisy[0] = (noisy[0] + math.pi) % (2 * math.pi) - math.pi
        noisy[1] = np.clip(noisy[1], 0.0, w.dist_max)
        endpoint = position + noisy[1] * np.array([math.cos(noisy[0]), math.sin(noisy[0])])
        if np.any(endpoint < 0) or endpoint[0] > w.length or endpoint[1] > w.width:
            return action
        if not w.transmit and np.linalg.norm(endpoint - w.target_loc) <= w.distance:
            return action
        # Guidance noise must not turn a safe planned step into an outage step.
        intended = position + action[1] * np.array([math.cos(action[0]), math.sin(action[0])])
        if w.get_g2a_outage_probability([endpoint]) >= 0.5 > w.get_g2a_outage_probability([intended]):
            return action
        return noisy

"""Turns NEAT genomes into driving policies and describes them for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import neat
import numpy as np


INPUT_LABELS = ("FRONT", "FRONT L", "FRONT R", "LEFT", "RIGHT", "REAR L", "REAR R", "SPEED", "HEADING")
OUTPUT_LABELS = ("STEER", "THROTTLE")


class NetworkController:
    """Drives every car with its own network. Decisions are never scripted."""

    def __init__(self, genomes: list, config: neat.Config):
        self.networks = [neat.nn.FeedForwardNetwork.create(g, config) for g in genomes]

    def __call__(self, ids: np.ndarray, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        steer = np.empty(ids.size)
        throttle = np.empty(ids.size)
        rows = observations.tolist()  # plain floats are much faster in neat's Python loop
        networks = self.networks
        for position, car in enumerate(ids.tolist()):
            steer[position], throttle[position] = networks[car].activate(rows[position])
        return steer, throttle

    def node_values(self, car: int) -> dict[int, float]:
        """Latest activation of each node in a car's network, for the live diagram."""
        return self.networks[car].values


@dataclass
class TopologyLayout:
    """Normalised positions of a network's nodes for drawing."""

    nodes: dict[int, tuple[float, float]]
    connections: list[tuple[int, int, float]]
    node_kinds: dict[int, str]
    hidden_count: int
    connection_count: int


def layout_topology(genome, config: neat.Config) -> TopologyLayout:
    """Places inputs on the left, outputs on the right, hidden nodes by network depth."""
    inputs = list(config.genome_config.input_keys)
    outputs = list(config.genome_config.output_keys)
    hidden = [key for key in genome.nodes if key not in outputs]
    connections = [(k[0], k[1], c.weight) for k, c in genome.connections.items() if c.enabled]

    # Depth of each node: the longest path to it from any input.
    depth = {key: 0 for key in inputs}
    remaining = set(hidden) | set(outputs)
    for _ in range(len(remaining) + 1):
        changed = False
        for source, target, _weight in connections:
            if source in depth and target in remaining:
                proposed = depth[source] + 1
                if proposed > depth.get(target, 0):
                    depth[target] = proposed
                    changed = True
        if not changed:
            break

    max_depth = max((depth.get(key, 1) for key in hidden), default=0) + 1

    positions: dict[int, tuple[float, float]] = {}
    kinds: dict[int, str] = {}
    for i, key in enumerate(inputs):
        positions[key] = (0.0, (i + 0.5) / len(inputs))
        kinds[key] = "input"
    for i, key in enumerate(outputs):
        positions[key] = (1.0, (i + 1) / (len(outputs) + 1))
        kinds[key] = "output"

    columns: dict[int, list[int]] = {}
    for key in hidden:
        columns.setdefault(max(depth.get(key, 1), 1), []).append(key)
    for level, keys in columns.items():
        for i, key in enumerate(sorted(keys)):
            positions[key] = (level / max_depth, (i + 0.5) / len(keys))
            kinds[key] = "hidden"

    return TopologyLayout(positions, connections, kinds, len(hidden), len(connections))

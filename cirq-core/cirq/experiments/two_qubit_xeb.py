# Copyright 2024 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence, TYPE_CHECKING, Optional, Tuple

import itertools
import functools

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

from cirq import ops, devices, value, vis
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
from cirq.experiments.xeb_fitting import benchmark_2q_xeb_fidelities
from cirq.experiments.xeb_fitting import fit_exponential_decays, exponential_decay
from cirq.experiments import random_quantum_circuit_generation as rqcg

if TYPE_CHECKING:
    import cirq


def _grid_qubits_for_sampler(sampler: 'cirq.Sampler'):
    if hasattr(sampler, 'processor'):
        device = sampler.processor.get_device()
        return sorted(device.metadata.qubit_set)
    else:
        qubits = devices.GridQubit.rect(3, 2, 4, 3)
        # Delete one qubit from the rectangular arangement to
        # 1) make it irregular 2) simplify simulation.
        return qubits[:-1]


def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
    return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)


def _decay_error(fidelity: float, n: int) -> float:
    return 1 - (1 - fidelity) * (2**n) / (2**n - 1)


class TwoQubitXEBResult:
    def __init__(self, fidelities: pd.DataFrame) -> None:
        self.fidelities = fidelities
        self._qubit_pair_map = {idx[-1]: i for i, idx in enumerate(fidelities.index)}

    def plot_heatmap(self, ax: Optional[plt.Axes] = None, target_error: str = 'xeb', **plot_kwargs):
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        error_func = self.xeb_error
        heatmap_data = {pair: error_func(*pair) for pair in self.all_qubit_pairs}

        ax.title(f'device {target_error} error heatmap')
        vis.TwoQubitInteractionHeatmap(heatmap_data).plot(ax=ax, **plot_kwargs)
        if show_plot:
            fig.show()

    def plot_fitted_exponential(
        self,
        q0: 'cirq.GridQubit',
        q1: 'cirq.GridQubit',
        ax: Optional[plt.Axes] = None,
        **plot_kwargs,
    ):
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if q0 > q1:
            q0, q1 = q1, q0
        record = self.fidelities.iloc[self._qubit_pair_map[(q0, q1)]]

        plt.axhline(1, color='grey', ls='--')
        plt.plot(record['cycle_depths'], record['fidelities'], 'o')
        xx = np.linspace(0, np.max(record['cycle_depths']))

        ax.plot(
            xx,
            exponential_decay(xx, a=record['a'], layer_fid=record['layer_fid']),
            label='estimated exponential decay',
            **plot_kwargs,
        )
        ax.title(f'{q0}-{q1}')
        ax.ylabel('Circuit fidelity')
        ax.xlabel('Cycle Depth $d$')
        ax.legend(loc='best')
        if show_plot:
            fig.show()

    def xeb_error(self, q0: 'cirq.GridQubit', q1: 'cirq.GridQubit'):
        if q0 > q1:
            q0, q1 = q1, q0
        p = self.fidelities.layer_fid[self._qubit_pair_map[(q0, q1)]]
        return 1 - p

    @functools.cached_property
    def all_qubit_pairs(self) -> frozenset[Tuple['cirq.GridQubit', 'Cirq.GridQubit']]:
        return frozenset(self._qubit_pair_map.keys())


def parallel_two_qubit_xeb(
    sampler: 'cirq.Sampler',
    entangling_gate: 'cirq.Gate' = ops.CZ,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = tuple(np.arange(3, 100, 20)),
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = 42,
) -> TwoQubitXEBResult:
    rs = value.parse_random_state(random_state)

    qubits = _grid_qubits_for_sampler(sampler)
    n_rows = max(q.row for q in qubits) + 1
    n_cols = max(q.col for q in qubits) * 2 + 1
    plt.figure(2, figsize=(n_cols, n_rows))
    graph = nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )
    nx.draw_networkx(graph, pos={q: (q.row, q.col) for q in qubits})
    plt.title('device layout')
    plt.show()

    print('Generate circuit library')
    circuit_library = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=n_circuits, two_qubit_gate=entangling_gate, random_state=rs
    )

    print('Generate random two qubit combinations')
    combs_by_layer = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuit_library),
        n_combinations=n_combinations,
        device_graph=graph,
        random_state=rs,
    )

    pos = {q: (q.row, q.col) for q in qubits}
    _, axes = plt.subplots(2, 2, figsize=(n_cols, n_rows))
    for comb_layer, ax in zip(combs_by_layer, axes.reshape(-1)):
        active_qubits = np.array(comb_layer.pairs).reshape(-1)
        colors = ['red' if q in active_qubits else 'blue' for q in graph.nodes]
        nx.draw_networkx(graph, pos=pos, node_color=colors, ax=ax)
        nx.draw_networkx_edges(
            graph, pos=pos, edgelist=comb_layer.pairs, width=3, edge_color='red', ax=ax
        )

    plt.tight_layout()

    print('Run circuits')
    sampled_df = sample_2q_xeb_circuits(
        sampler=sampler,
        circuits=circuit_library,
        cycle_depths=cycle_depths,
        combinations_by_layer=combs_by_layer,
        shuffle=rs,
        repetitions=n_repetitions,
    )

    print('Compute fidelities')
    fids = benchmark_2q_xeb_fidelities(
        sampled_df=sampled_df, circuits=circuit_library, cycle_depths=cycle_depths
    )

    print('Fit exponential decays')
    return TwoQubitXEBResult(fit_exponential_decays(fids))

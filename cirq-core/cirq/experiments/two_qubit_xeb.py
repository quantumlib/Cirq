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
from typing import Sequence, TYPE_CHECKING

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

from cirq import ops, devices, value, vis
import cirq.contrib.routing as ccr
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
from cirq.experiments.xeb_fitting import benchmark_2q_xeb_fidelities
from cirq.experiments.xeb_fitting import fit_exponential_decays, exponential_decay
from cirq.experiments import random_quantum_circuit_generation as rqcg

if TYPE_CHECKING:
    import cirq


def grid_qubits_for_sampler(sampler: 'cirq.Sampler'):
    if hasattr(sampler, 'processor'):
        device = sampler.processor.get_device()
        return sorted(device.metadata.qubit_set)
    else:
        qubits = devices.GridQubit.rect(3, 2, 4, 3)
        # Delete one qubit from the rectangular arangement to
        # 1) make it irregular 2) simplify simulation.
        return qubits[:-1]


def parallel_two_qubit_randomized_benchmarking(
    sampler: 'cirq.Sampler',
    entangling_gate: 'cirq.Gate' = ops.CZ,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = tuple(np.arange(3, 100, 20)),
    random_state: 'value.RANDOM_STATE_OR_SEED_LIKE' = 42,
):
    rs = value.parse_random_state(random_state)

    qubits = grid_qubits_for_sampler(sampler)
    n_rows = max(q.row for q in qubits) * 2 + 1
    n_cols = max(q.col for q in qubits) * 2 + 1
    plt.figure(2, figsize=(n_cols, n_rows))
    graph = ccr.gridqubits_to_graph_device(qubits)
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
    return fit_exponential_decays(fids)


def visulaize_fidelities(fidelities: pd.DataFrame):
    heatmap_data = {}
    for (_, _, pair), fidelity in fidelities.layer_fid.items():
        heatmap_data[pair] = 1.0 - fidelity

    vis.TwoQubitInteractionHeatmap(heatmap_data).plot()
    plt.title('device fidelity heatmap')
    plt.show()

    for i, record in fidelities.iterrows():
        plt.axhline(1, color='grey', ls='--')
        plt.plot(record['cycle_depths'], record['fidelities'], 'o')
        xx = np.linspace(0, np.max(record['cycle_depths']))
        plt.plot(
            xx,
            exponential_decay(xx, a=record['a'], layer_fid=record['layer_fid']),
            label='estimated exponential decay',
        )
        q0, q1 = i[-1]
        plt.title(f'{q0}-{q1}')
        plt.ylabel('Circuit fidelity')
        plt.xlabel('Cycle Depth $d$')
        plt.legend(loc='best')
        plt.show()
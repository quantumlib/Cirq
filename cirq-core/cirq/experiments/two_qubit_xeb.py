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
from typing import Sequence, TYPE_CHECKING, Optional, Tuple, Dict

from dataclasses import dataclass
import itertools
import functools

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

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


def _manhattan_distance(qubit1: 'cirq.GridQubit', qubit2: 'cirq.GridQubit') -> int:
    return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)


@dataclass(frozen=True)
class TwoQubitXEBResult:
    """Results from an XEB experiment."""

    fidelities: pd.DataFrame

    @functools.cached_property
    def _qubit_pair_map(self) -> Dict[Tuple['cirq.GridQubit', 'cirq.GridQubit'], int]:
        return {
            (min(q0, q1), max(q0, q1)): i
            for i, (_, _, (q0, q1)) in enumerate(self.fidelities.index)
        }

    @functools.cached_property
    def all_qubit_pairs(self) -> Tuple[Tuple['cirq.GridQubit', 'cirq.GridQubit'], ...]:
        return tuple(sorted(self._qubit_pair_map.keys()))

    def plot_heatmap(self, ax: Optional[plt.Axes] = None, **plot_kwargs) -> plt.Axes:
        """plot the heatmap for xeb error.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        """
        show_plot = not ax
        if not isinstance(ax, plt.Axes):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        heatmap_data: Dict[Tuple['cirq.GridQubit', ...], float] = {
            pair: self.xeb_error(*pair) for pair in self.all_qubit_pairs
        }

        ax.set_title('device xeb error heatmap')

        vis.TwoQubitInteractionHeatmap(heatmap_data).plot(ax=ax, **plot_kwargs)
        if show_plot:
            fig.show()
        return ax

    def plot_fitted_exponential(
        self,
        q0: 'cirq.GridQubit',
        q1: 'cirq.GridQubit',
        ax: Optional[plt.Axes] = None,
        **plot_kwargs,
    ) -> plt.Axes:
        """plot the fitted model to for xeb error of a qubit pair.

        Args:
            q0: first qubit.
            q1: second qubit.
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        """
        show_plot = not ax
        if not isinstance(ax, plt.Axes):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        record = self._record(q0, q1)

        ax.axhline(1, color='grey', ls='--')
        ax.plot(record['cycle_depths'], record['fidelities'], 'o')
        depths = np.linspace(0, np.max(record['cycle_depths']))
        ax.plot(
            depths,
            exponential_decay(depths, a=record['a'], layer_fid=record['layer_fid']),
            label='estimated exponential decay',
            **plot_kwargs,
        )
        ax.set_title(f'{q0}-{q1}')
        ax.set_ylabel('Circuit fidelity')
        ax.set_xlabel('Cycle Depth $d$')
        ax.legend(loc='best')
        if show_plot:
            fig.show()
        return ax

    def _record(self, q0, q1) -> pd.Series:
        if q0 > q1:
            q0, q1 = q1, q0
        return self.fidelities.iloc[self._qubit_pair_map[(q0, q1)]]

    def xeb_error(self, q0: 'cirq.GridQubit', q1: 'cirq.GridQubit') -> float:
        """Return the XEB error of a qubit pair."""
        p = self._record(q0, q1).layer_fid
        return 1 - p

    def all_errors(self) -> Dict[Tuple['cirq.GridQubit', 'cirq.GridQubit'], float]:
        """Return the XEB error of all qubit pairs."""
        return {(q0, q1): self.xeb_error(q0, q1) for q0, q1 in self.all_qubit_pairs}

    def plot_histogram(self, ax: Optional[plt.Axes] = None, **plot_kwargs) -> plt.Axes:
        """plot a histogram of all xeb errors

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        vis.integrated_histogram(data=self.all_errors(), ax=ax, **plot_kwargs)
        if fig is not None:
            fig.show(**plot_kwargs)
        return ax


def parallel_two_qubit_xeb(
    sampler: 'cirq.Sampler',
    entangling_gate: 'cirq.Gate' = ops.CZ,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = tuple(np.arange(3, 100, 20)),
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = 42,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> TwoQubitXEBResult:
    """A convenience method that runs the full XEB workflow.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        entangling_gate: The entangling gate to use.
        n_repetitions: The number of repetitions to use.
        n_combinations: The number of combinations to generate.
        n_circuits: The number of circuits to generate.
        cycle_depths: The cycle depths to use.
        random_state: The random state to use.
        ax: the plt.Axes to plot the device layout on. If not given,
            no plot is created.
        **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

    Returns:
        A TwoQubitXEBResult object representing the results of the experiment.
    """
    rs = value.parse_random_state(random_state)

    qubits = _grid_qubits_for_sampler(sampler)
    graph = nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )

    if ax is not None:
        nx.draw_networkx(graph, pos={q: (q.row, q.col) for q in qubits}, ax=ax)
        ax.set_title('device layout')
        ax.plot(**plot_kwargs)

    circuit_library = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=n_circuits, two_qubit_gate=entangling_gate, random_state=rs
    )

    combs_by_layer = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuit_library),
        n_combinations=n_combinations,
        device_graph=graph,
        random_state=rs,
    )

    sampled_df = sample_2q_xeb_circuits(
        sampler=sampler,
        circuits=circuit_library,
        cycle_depths=cycle_depths,
        combinations_by_layer=combs_by_layer,
        shuffle=rs,
        repetitions=n_repetitions,
    )

    fids = benchmark_2q_xeb_fidelities(
        sampled_df=sampled_df, circuits=circuit_library, cycle_depths=cycle_depths
    )

    return TwoQubitXEBResult(fit_exponential_decays(fids))

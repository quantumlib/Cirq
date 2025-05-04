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

"""Provides functions for running and analyzing two-qubit XEB experiments."""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast, Dict, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cirq import ops, value, vis
from cirq._compat import cached_method
from cirq.experiments import random_quantum_circuit_generation as rqcg
from cirq.experiments.qubit_characterizations import (
    parallel_single_qubit_randomized_benchmarking,
    ParallelRandomizedBenchmarkingResult,
)
from cirq.experiments.xeb_fitting import (
    benchmark_2q_xeb_fidelities,
    exponential_decay,
    fit_exponential_decays,
)
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
from cirq.qis import noise_utils

if TYPE_CHECKING:
    import multiprocessing

    import cirq


def _grid_qubits_for_sampler(sampler: cirq.Sampler) -> Optional[Sequence[cirq.GridQubit]]:
    if hasattr(sampler, 'processor'):
        device = sampler.processor.get_device()
        return sorted(device.metadata.qubit_set)
    return None


def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
    return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)


def qubits_and_pairs(
    sampler: cirq.Sampler,
    qubits: Optional[Sequence[cirq.GridQubit]] = None,
    pairs: Optional[Sequence[tuple[cirq.GridQubit, cirq.GridQubit]]] = None,
) -> Tuple[Sequence[cirq.GridQubit], Sequence[tuple[cirq.GridQubit, cirq.GridQubit]]]:
    """Extract qubits and pairs from sampler.


    If qubits are not provided, then they are extracted from the pairs (if given) or the sampler.
    If pairs are not provided then all pairs of adjacent qubits are used.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: Optional list of qubits.
        pairs: Optional list of pair to use.

    Returns:
        - Qubits to use.
        - Pairs of qubits to use.

    Raises:
        ValueError: If qubits are not specified and can't be deduced from other arguments.
    """
    if qubits is None:
        if pairs is None:
            qubits = _grid_qubits_for_sampler(sampler)
            if qubits is None:
                raise ValueError("Couldn't determine qubits from sampler. Please specify them.")
        else:
            qubits_set = set(itertools.chain(*pairs))
            qubits = list(qubits_set)

    if pairs is None:
        pairs = [
            pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
        ]

    return qubits, pairs


@dataclass(frozen=True)
class TwoQubitXEBResult:
    """Results from an XEB experiment."""

    fidelities: pd.DataFrame

    @functools.cached_property
    def _qubit_pair_map(self) -> Dict[Tuple[cirq.GridQubit, cirq.GridQubit], int]:
        if isinstance(self.fidelities.index[0][0], ops.Qid):
            return {
                (min(q0, q1), max(q0, q1)): i for i, (q0, q1) in enumerate(self.fidelities.index)
            }
        return {
            (min(q0, q1), max(q0, q1)): i
            for i, (_, _, (q0, q1)) in enumerate(self.fidelities.index)
        }

    @functools.cached_property
    def all_qubit_pairs(self) -> Tuple[Tuple[cirq.GridQubit, cirq.GridQubit], ...]:
        return tuple(sorted(self._qubit_pair_map.keys()))

    def plot_heatmap(self, ax: Optional[plt.Axes] = None, **plot_kwargs) -> plt.Axes:
        """plot the heatmap of XEB errors.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes that was plotted on.
        """
        show_plot = not ax
        if not isinstance(ax, plt.Axes):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        heatmap_data: Dict[Tuple[cirq.GridQubit, ...], float] = {
            pair: self.xeb_error(*pair) for pair in self.all_qubit_pairs
        }

        ax.set_title('device xeb error heatmap')

        vis.TwoQubitInteractionHeatmap(heatmap_data).plot(ax=ax, **plot_kwargs)
        if show_plot:
            fig.show()
        return ax

    def plot_fitted_exponential(
        self, q0: cirq.GridQubit, q1: cirq.GridQubit, ax: Optional[plt.Axes] = None, **plot_kwargs
    ) -> plt.Axes:
        """plot the fitted model to for xeb error of a qubit pair.

        Args:
            q0: first qubit.
            q1: second qubit.
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes that was plotted on.
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

    def xeb_fidelity(self, q0: cirq.GridQubit, q1: cirq.GridQubit) -> float:
        """Return the XEB fidelity of a qubit pair."""
        return noise_utils.decay_constant_to_xeb_fidelity(
            self._record(q0, q1).layer_fid, num_qubits=2
        )

    def xeb_error(self, q0: cirq.GridQubit, q1: cirq.GridQubit) -> float:
        """Return the XEB error of a qubit pair."""
        return 1 - self.xeb_fidelity(q0, q1)

    def all_errors(self) -> Dict[Tuple[cirq.GridQubit, cirq.GridQubit], float]:
        """Return the XEB error of all qubit pairs."""
        return {(q0, q1): self.xeb_error(q0, q1) for q0, q1 in self.all_qubit_pairs}

    def plot_histogram(self, ax: Optional[plt.Axes] = None, **plot_kwargs) -> plt.Axes:
        """plot a histogram of all xeb errors.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes that was plotted on.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        vis.integrated_histogram(data=self.all_errors(), ax=ax, **plot_kwargs)
        if fig is not None:
            fig.show(**plot_kwargs)
        return ax

    @cached_method
    def pauli_error(self) -> Dict[Tuple[cirq.GridQubit, cirq.GridQubit], float]:
        """Return the Pauli error of all qubit pairs."""
        return {
            pair: noise_utils.decay_constant_to_pauli_error(
                self._record(*pair).layer_fid, num_qubits=2
            )
            for pair in self.all_qubit_pairs
        }


@dataclass(frozen=True)
class InferredXEBResult:
    """Uses the results from XEB and RB to compute inferred two-qubit Pauli errors.

    The result of running just XEB combines both two-qubit and single-qubit error rates,
    this class computes inferred errors which are the result of removing the single qubit errors
    from the two-qubit errors.
    """

    rb_result: ParallelRandomizedBenchmarkingResult
    xeb_result: TwoQubitXEBResult

    @property
    def all_qubit_pairs(self) -> Sequence[Tuple[cirq.GridQubit, cirq.GridQubit]]:
        return self.xeb_result.all_qubit_pairs

    @cached_method
    def single_qubit_pauli_error(self) -> Mapping[cirq.Qid, float]:
        """Return the single-qubit Pauli error for all qubits (RB results)."""
        return self.rb_result.pauli_error()

    @cached_method
    def two_qubit_pauli_error(self) -> Mapping[Tuple[cirq.GridQubit, cirq.GridQubit], float]:
        """Return the two-qubit Pauli error for all pairs."""
        return MappingProxyType(self.xeb_result.pauli_error())

    @cached_method
    def inferred_pauli_error(self) -> Mapping[Tuple[cirq.GridQubit, cirq.GridQubit], float]:
        """Return the inferred Pauli error for all pairs."""
        single_q_paulis = self.rb_result.pauli_error()
        xeb = self.xeb_result.pauli_error()

        def _pauli_error(q0: cirq.GridQubit, q1: cirq.GridQubit) -> float:
            q0, q1 = sorted([q0, q1])
            return xeb[(q0, q1)] - single_q_paulis[q0] - single_q_paulis[q1]

        return MappingProxyType({pair: _pauli_error(*pair) for pair in self.all_qubit_pairs})

    @cached_method
    def inferred_decay_constant(self) -> Mapping[Tuple[cirq.GridQubit, cirq.GridQubit], float]:
        """Return the inferred decay constant for all pairs."""
        return MappingProxyType(
            {
                pair: noise_utils.pauli_error_to_decay_constant(pauli, 2)
                for pair, pauli in self.inferred_pauli_error().items()
            }
        )

    @cached_method
    def inferred_xeb_error(self) -> Mapping[Tuple[cirq.GridQubit, cirq.GridQubit], float]:
        """Return the inferred XEB error for all pairs."""
        return MappingProxyType(
            {
                pair: 1 - noise_utils.decay_constant_to_xeb_fidelity(decay, 2)
                for pair, decay in self.inferred_decay_constant().items()
            }
        )

    def _target_errors(
        self, target_error: str
    ) -> Mapping[Tuple[cirq.GridQubit, cirq.GridQubit], float]:
        """Returns requested error.

        The requested error must be one of 'pauli', 'decay_constant', or 'xeb'.

        Args:
            target_error: The error to draw.

        Returns:
            A mapping of qubit pairs to the requested error.

        Raises:
            ValueError: If the requested error is not one of 'pauli', 'decay_constant', or 'xeb'.
        """
        error_funcs = {
            'pauli': self.inferred_pauli_error,
            'decay_constant': self.inferred_decay_constant,
            'xeb': self.inferred_xeb_error,
        }
        return error_funcs[target_error]()

    def plot_heatmap(
        self, target_error: str = 'pauli', ax: Optional[plt.Axes] = None, **plot_kwargs
    ) -> plt.Axes:
        """plot the heatmap of the target errors.

        Args:
            target_error: The error to draw. Must be one of 'xeb', 'pauli', or 'decay_constant'
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        """
        show_plot = not ax
        if not isinstance(ax, plt.Axes):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        heatmap_data = cast(
            Mapping[Tuple['cirq.GridQubit', ...], float], self._target_errors(target_error)
        )

        name = f'{target_error} error' if target_error != 'decay_constant' else 'decay constant'
        ax.set_title(f'device {name} heatmap')

        vis.TwoQubitInteractionHeatmap(heatmap_data).plot(ax=ax, **plot_kwargs)
        if show_plot:
            fig.show()
        return ax

    def plot_histogram(
        self,
        target_error: str = 'pauli',
        ax: Optional[plt.Axes] = None,
        kind: str = 'two_qubit',
        **plot_kwargs,
    ) -> plt.Axes:
        """plot a histogram of target error.

        Args:
            target_error: The error to draw. Must be one of 'xeb', 'pauli', or 'decay_constant'
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            kind: Whether to plot the single-qubit RB errors ('single_qubit') or the
                two-qubit inferred errors ('two_qubit') or both ('both').
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes that was plotted on.

        Raises:
            ValueError: If
                - `kind` is not one of 'single_qubit', 'two_qubit', or 'both'.
                - `target_error` is not one of 'pauli', 'xeb', or 'decay_constant'
                - single qubit error is requested and `target_error` is not 'pauli'.
        """
        if kind not in ('single_qubit', 'two_qubit', 'both'):
            raise ValueError(
                f"kind must be one of 'single_qubit', 'two_qubit', or 'both', not {kind}"
            )
        if kind != 'two_qubit' and target_error != 'pauli':
            raise ValueError(f'{target_error} is not supported for single qubits')
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        alpha = 0.5 if kind == 'both' else 1.0
        if kind == 'single_qubit' or kind == 'both':
            self.rb_result.plot_integrated_histogram(
                ax=ax, alpha=alpha, label='single qubit', color='green', **plot_kwargs
            )
        if kind == 'two_qubit' or kind == 'both':
            vis.integrated_histogram(
                data=self._target_errors(target_error),
                ax=ax,
                alpha=alpha,
                label='two qubit',
                color='blue',
                **plot_kwargs,
            )

        if fig is not None:
            fig.show(**plot_kwargs)
        return ax


def parallel_xeb_workflow(
    sampler: cirq.Sampler,
    qubits: Optional[Sequence[cirq.GridQubit]] = None,
    entangling_gate: cirq.Gate = ops.CZ,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = (5, 25, 50, 100, 200, 300),
    random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ax: Optional[plt.Axes] = None,
    pairs: Optional[Sequence[tuple[cirq.GridQubit, cirq.GridQubit]]] = None,
    pool: Optional[multiprocessing.pool.Pool] = None,
    batch_size: int = 9,
    tags: Sequence[Any] = (),
    **plot_kwargs,
) -> Tuple[pd.DataFrame, Sequence[cirq.Circuit], pd.DataFrame]:
    """A utility method that runs the full XEB workflow.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: Qubits under test. If none, uses all qubits on the sampler's device.
        entangling_gate: The entangling gate to use.
        n_repetitions: The number of repetitions to use.
        n_combinations: The number of combinations to generate.
        n_circuits: The number of circuits to generate.
        cycle_depths: The cycle depths to use.
        random_state: The random state to use.
        ax: the plt.Axes to plot the device layout on. If not given,
            no plot is created.
        pairs: Pairs to use. If not specified, use all pairs between adjacent qubits.
        pool: An optional multiprocessing pool.
        batch_size: We call `run_batch` on the sampler, which can speed up execution in certain
            environments. The number of (circuit, cycle_depth) tasks to be run in each batch
            is given by this number.
        tags: Tags to add to two qubit operations.
        **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

    Returns:
        - A DataFrame with columns 'cycle_depth' and 'fidelity'.
        - The circuits used to perform XEB.
        - A pandas dataframe with index given by ['circuit_i', 'cycle_depth'].
            Columns always include "sampled_probs". If `combinations_by_layer` is
            not `None` and you are doing parallel XEB, additional metadata columns
            will be attached to the returned DataFrame.

    Raises:
        ValueError: If qubits are not specified and the sampler has no device.
    """
    rs = value.parse_random_state(random_state)

    qubits, pairs = qubits_and_pairs(sampler, qubits, pairs)
    graph = nx.Graph(pairs)

    if ax is not None:
        nx.draw_networkx(graph, pos={q: (q.row, q.col) for q in qubits}, ax=ax)
        ax.set_title('device layout')
        ax.plot(**plot_kwargs)

    circuit_library = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=n_circuits,
        two_qubit_gate=entangling_gate,
        random_state=rs,
        max_cycle_depth=max(cycle_depths),
        tags=tags,
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
        batch_size=batch_size,
    )

    fids = benchmark_2q_xeb_fidelities(
        sampled_df=sampled_df, circuits=circuit_library, cycle_depths=cycle_depths, pool=pool
    )

    return fids, circuit_library, sampled_df


def parallel_two_qubit_xeb(
    sampler: cirq.Sampler,
    qubits: Optional[Sequence[cirq.GridQubit]] = None,
    entangling_gate: cirq.Gate = ops.CZ,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = (5, 25, 50, 100, 200, 300),
    random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ax: Optional[plt.Axes] = None,
    pairs: Optional[Sequence[tuple[cirq.GridQubit, cirq.GridQubit]]] = None,
    batch_size: int = 9,
    tags: Sequence[Any] = (),
    **plot_kwargs,
) -> TwoQubitXEBResult:
    """A convenience method that runs the full XEB workflow.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: Qubits under test. If none, uses all qubits on the sampler's device.
        entangling_gate: The entangling gate to use.
        n_repetitions: The number of repetitions to use.
        n_combinations: The number of combinations to generate.
        n_circuits: The number of circuits to generate.
        cycle_depths: The cycle depths to use.
        random_state: The random state to use.
        ax: the plt.Axes to plot the device layout on. If not given,
            no plot is created.
        pairs: Pairs to use. If not specified, use all pairs between adjacent qubits.
        batch_size: We call `run_batch` on the sampler, which can speed up execution in certain
            environments. The number of (circuit, cycle_depth) tasks to be run in each batch
            is given by this number.
        tags: Tags to add to two qubit operations.
        **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
    Returns:
        A TwoQubitXEBResult object representing the results of the experiment.
    Raises:
        ValueError: If qubits are not specified and the sampler has no device.
    """
    fids, *_ = parallel_xeb_workflow(
        sampler=sampler,
        qubits=qubits,
        pairs=pairs,
        entangling_gate=entangling_gate,
        n_repetitions=n_repetitions,
        n_combinations=n_combinations,
        n_circuits=n_circuits,
        cycle_depths=cycle_depths,
        random_state=random_state,
        ax=ax,
        batch_size=batch_size,
        tags=tags,
        **plot_kwargs,
    )
    return TwoQubitXEBResult(fit_exponential_decays(fids))


def run_rb_and_xeb(
    sampler: cirq.Sampler,
    qubits: Optional[Sequence[cirq.GridQubit]] = None,
    repetitions: int = 10**3,
    num_circuits: int = 20,
    num_clifford_range: Sequence[int] = tuple(
        np.logspace(np.log10(5), np.log10(1000), 5, dtype=int)
    ),
    entangling_gate: cirq.Gate = ops.CZ,
    depths_xeb: Sequence[int] = (5, 25, 50, 100, 200, 300),
    xeb_combinations: int = 10,
    random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    pairs: Optional[Sequence[tuple[cirq.GridQubit, cirq.GridQubit]]] = None,
    batch_size: int = 9,
    tags: Sequence[Any] = (),
) -> InferredXEBResult:
    """A convenience method that runs both RB and XEB workflows.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: Qubits under test. If none, uses all qubits on the sampler's device.
        repetitions: The number of repetitions to use for RB and XEB.
        num_circuits: The number of circuits to generate for RB and XEB.
        num_clifford_range: The different numbers of Cliffords in the RB study.
        entangling_gate: The entangling gate to use.
        depths_xeb: The cycle depths to use for XEB.
        xeb_combinations: The number of combinations to generate for XEB.
        random_state: The random state to use.
        pairs: Pairs to use. If not specified, use all pairs between adjacent qubits.
        batch_size: We call `run_batch` on the sampler, which can speed up execution in certain
            environments. The number of (circuit, cycle_depth) tasks to be run in each batch
            is given by this number.
        tags: Tags to add to two qubit operations.

    Returns:
        An InferredXEBResult object representing the results of the experiment.

    Raises:
        ValueError: If qubits are not specified and the sampler has no device.
    """

    qubits, pairs = qubits_and_pairs(sampler, qubits, pairs)

    rb = parallel_single_qubit_randomized_benchmarking(
        sampler=sampler,
        qubits=qubits,
        repetitions=repetitions,
        num_circuits=num_circuits,
        num_clifford_range=num_clifford_range,
    )

    xeb = parallel_two_qubit_xeb(
        sampler=sampler,
        qubits=qubits,
        pairs=pairs,
        entangling_gate=entangling_gate,
        n_repetitions=repetitions,
        n_circuits=num_circuits,
        cycle_depths=depths_xeb,
        n_combinations=xeb_combinations,
        random_state=random_state,
        batch_size=batch_size,
        tags=tags,
    )

    return InferredXEBResult(rb, xeb)

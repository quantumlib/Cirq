# Copyright 2019 The Cirq Developers
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
"""Simulator for density matrices that simulates noisy quantum circuits."""
from typing import Any, Dict, TYPE_CHECKING, Tuple, Union, Sequence, Optional, List

import numpy as np

from cirq import ops, protocols, qis, study, value
from cirq.sim import (
    simulator,
    act_on_density_matrix_args,
    simulator_base,
)

if TYPE_CHECKING:
    import cirq
    from numpy.typing import DTypeLike


class DensityMatrixSimulator(
    simulator_base.SimulatorBase[
        'DensityMatrixStepResult',
        'DensityMatrixTrialResult',
        'DensityMatrixSimulatorState',
        act_on_density_matrix_args.ActOnDensityMatrixArgs,
    ],
    simulator.SimulatesExpectationValues,
):
    """A simulator for density matrices and noisy quantum circuits.

    This simulator can be applied on circuits that are made up of operations
    that have:
        * a `_kraus_` method for a Kraus representation of a quantum channel.
        * a `_mixture_` method for a probabilistic combination of unitary gates.
        * a `_unitary_` method for a unitary gate.
        * a `_has_unitary_` and `_apply_unitary_` method.
        * measurements
        * a `_decompose_` that eventually yields one of the above
    That is, the circuit must have elements that follow on of the protocols:
        * `cirq.SupportsKraus`
        * `cirq.SupportsMixture`
        * `cirq.SupportsConsistentApplyUnitary`
        * `cirq.SupportsUnitary`
        * `cirq.SupportsDecompose`
    or is a measurement.

    This simulator supports three types of simulation.

    Run simulations which mimic running on actual quantum hardware. These
    simulations do not give access to the density matrix (like actual hardware).
    There are two variations of run methods, one which takes in a single
    (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolver:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    These methods return `Result`s which contain both the measurement
    results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast the simulate methods of the simulator give access to the density
    matrix of the simulation at the end of the simulation of the circuit.
    Note that if the circuit contains measurements then the density matrix
    is that result for those particular measurement results. For example
    if there is one measurement, then the simulation may result in the
    measurement result for this measurement, and the density matrix will
    be that conditional on that result. It will not be the density matrix formed
    by summing over the different measurements and their probabilities.
    The simulate methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `DensityMatrixTrialResult` for details of this ordering). The initial
    state can be either the full density matrix, the full wave function (for
    pure states), or an integer which represents the initial state of being
    in a computational basis state for the binary representation of that
    integer. Similar to run methods, there are two simulate methods that run
    for single simulations or for sweeps across different parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods in contrast to the run methods do not perform
    repetitions. The result of these simulations is a
    `DensityMatrixTrialResult` which contains, in addition to measurement
    results and information about the parameters that were used in the
    simulation, access to the density matrix via the `density_matrix` method.

    If one wishes to perform simulations that have access to the
    density matrix as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the density matrix.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments via

        for step_result in simulate_moments(circuit):
           # do something with the density matrix via
           # step_result.density_matrix()
    """

    def __init__(
        self,
        *,
        dtype: 'DTypeLike' = np.complex64,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        ignore_measurement_results: bool = False,
        split_untangled_states: bool = True,
    ):
        """Density matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            ignore_measurement_results: if True, then the simulation
                will treat measurement as dephasing instead of collapsing
                process.
            split_untangled_states: If True, optimizes simulation by running
                unentangled qubit sets independently and merging those states
                at the end.

        Raises:
            ValueError: If the supplied dtype is not `np.complex64` or
                `np.complex128`.

        Example:
           >>> (q0,) = cirq.LineQubit.range(1)
           >>> circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

           Default case (ignore_measurement_results = False):
           >>> simulator = cirq.DensityMatrixSimulator()
           >>> result = simulator.run(circuit)

           The measurement result will be strictly one of 0 or 1.

           In the other case:
           >>> simulator = cirq.DensityMatrixSimulator(
           ...     ignore_measurement_results = True)

           Will raise a `ValueError` exception if you call `simulator.run`
           when `ignore_measurement_results` has been set to True
           (for more see https://github.com/quantumlib/Cirq/issues/2777).
        """
        super().__init__(
            dtype=dtype,
            noise=noise,
            seed=seed,
            ignore_measurement_results=ignore_measurement_results,
            split_untangled_states=split_untangled_states,
        )
        if dtype not in {np.complex64, np.complex128}:
            raise ValueError(f'dtype must be complex64 or complex128, was {dtype}')

    def _create_partial_act_on_args(
        self,
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE', 'cirq.ActOnDensityMatrixArgs'],
        qubits: Sequence['cirq.Qid'],
        logs: Dict[str, Any],
    ) -> 'cirq.ActOnDensityMatrixArgs':
        """Creates the ActOnDensityMatrixArgs for a circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            logs: The log of measurement results that is added into.

        Returns:
            ActOnDensityMatrixArgs for the circuit.
        """
        if isinstance(initial_state, act_on_density_matrix_args.ActOnDensityMatrixArgs):
            return initial_state

        qid_shape = protocols.qid_shape(qubits)
        initial_matrix = qis.to_valid_density_matrix(
            initial_state, len(qid_shape), qid_shape=qid_shape, dtype=self._dtype
        )
        if np.may_share_memory(initial_matrix, initial_state):
            initial_matrix = initial_matrix.copy()

        tensor = initial_matrix.reshape(qid_shape * 2)
        return act_on_density_matrix_args.ActOnDensityMatrixArgs(
            target_tensor=tensor,
            available_buffer=[np.empty_like(tensor) for _ in range(3)],
            qubits=qubits,
            qid_shape=qid_shape,
            prng=self._prng,
            log_of_measurement_results=logs,
        )

    def _can_be_in_run_prefix(self, val: Any):
        return not protocols.measurement_keys_touched(val)

    def _create_step_result(
        self,
        sim_state: 'cirq.OperationTarget[cirq.ActOnDensityMatrixArgs]',
    ):
        return DensityMatrixStepResult(
            sim_state=sim_state,
            simulator=self,
            dtype=self._dtype,
        )

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_step_result: 'DensityMatrixStepResult',
    ) -> 'DensityMatrixTrialResult':
        return DensityMatrixTrialResult(
            params=params, measurements=measurements, final_step_result=final_step_result
        )

    # TODO(#4209): Deduplicate with identical code in sparse_simulator.
    def simulate_expectation_values_sweep(
        self,
        program: 'cirq.AbstractCircuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        params: 'study.Sweepable',
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> List[List[float]]:
        if not permit_terminal_measurements and program.are_any_measurements_terminal():
            raise ValueError(
                'Provided circuit has terminal measurements, which may '
                'skew expectation values. If this is intentional, set '
                'permit_terminal_measurements=True.'
            )
        swept_evs = []
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qmap = {q: i for i, q in enumerate(qubit_order.order_for(program.all_qubits()))}
        if not isinstance(observables, List):
            observables = [observables]
        pslist = [ops.PauliSum.wrap(pslike) for pslike in observables]
        for param_resolver in study.to_resolvers(params):
            result = self.simulate(
                program, param_resolver, qubit_order=qubit_order, initial_state=initial_state
            )
            swept_evs.append(
                [
                    obs.expectation_from_density_matrix(result.final_density_matrix, qmap)
                    for obs in pslist
                ]
            )
        return swept_evs


class DensityMatrixStepResult(
    simulator_base.StepResultBase[
        'DensityMatrixSimulatorState', act_on_density_matrix_args.ActOnDensityMatrixArgs
    ]
):
    """A single step in the simulation of the DensityMatrixSimulator.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(
        self,
        sim_state: 'cirq.OperationTarget[cirq.ActOnDensityMatrixArgs]',
        simulator: DensityMatrixSimulator,
        dtype: 'DTypeLike' = np.complex64,
    ):
        """DensityMatrixStepResult.

        Args:
            sim_state: The qubit:ActOnArgs lookup for this step.
            simulator: The simulator used to create this.
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
        """
        super().__init__(sim_state)
        self._dtype = dtype
        self._density_matrix: Optional[np.ndarray] = None
        self._simulator = simulator

    def _simulator_state(self) -> 'DensityMatrixSimulatorState':
        return DensityMatrixSimulatorState(self.density_matrix(copy=False), self._qubit_mapping)

    def set_density_matrix(self, density_matrix_repr: Union[int, np.ndarray]):
        """Set the density matrix to a new density matrix.

        Args:
            density_matrix_repr: If this is an int, the density matrix is set to
            the computational basis state corresponding to this state. Otherwise
            if this is a np.ndarray it is the full state, either a pure state
            or the full density matrix.  If it is the pure state it must be the
            correct size, be normalized (an L2 norm of 1), and be safely
            castable to an appropriate dtype for the simulator.  If it is a
            mixed state it must be correctly sized and positive semidefinite
            with trace one.
        """
        self._sim_state = self._simulator._create_act_on_args(density_matrix_repr, self._qubits)

    def density_matrix(self, copy=True):
        """Returns the density matrix at this step in the simulation.

        The density matrix that is stored in this result is returned in the
        computational basis with these basis states defined by the qubit_map.
        In particular the value in the qubit_map is the index of the qubit,
        and these are translated into binary vectors where the last qubit is
        the 1s bit of the index, the second-to-last is the 2s bit of the index,
        and so forth (i.e. big endian ordering). The density matrix is a
        `2 ** num_qubits` square matrix, with rows and columns ordered by
        the computational basis as just described.

        Example:
             qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
             Then the returned density matrix will have (row and column) indices
             mapped to qubit basis states like the following table

                |     | QubitA | QubitB | QubitC |
                | :-: | :----: | :----: | :----: |
                |  0  |   0    |   0    |   0    |
                |  1  |   0    |   0    |   1    |
                |  2  |   0    |   1    |   0    |
                |  3  |   0    |   1    |   1    |
                |  4  |   1    |   0    |   0    |
                |  5  |   1    |   0    |   1    |
                |  6  |   1    |   1    |   0    |
                |  7  |   1    |   1    |   1    |

        Args:
            copy: If True, then the returned state is a copy of the density
                matrix. If False, then the density matrix is not copied,
                potentially saving memory. If one only needs to read derived
                parameters from the density matrix and store then using False
                can speed up simulation by eliminating a memory copy.
        """
        if self._density_matrix is None:
            self._density_matrix = np.array(1)
            state = self._merged_sim_state
            if state is not None:
                matrix = state.target_tensor
                size = int(np.sqrt(np.prod(matrix.shape, dtype=np.int64)))
                self._density_matrix = np.reshape(matrix, (size, size))
        return self._density_matrix.copy() if copy else self._density_matrix


@value.value_equality(unhashable=True)
class DensityMatrixSimulatorState:
    """The simulator state for DensityMatrixSimulator

    Args:
        density_matrix: The density matrix of the simulation.
        qubit_map: A map from qid to index used to define the
            ordering of the basis in density_matrix.
    """

    def __init__(self, density_matrix: np.ndarray, qubit_map: Dict[ops.Qid, int]) -> None:
        self.density_matrix = density_matrix
        self.qubit_map = qubit_map
        self._qid_shape = simulator._qubit_map_to_shape(qubit_map)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _value_equality_values_(self) -> Any:
        return (self.density_matrix.tolist(), self.qubit_map)

    def __repr__(self) -> str:
        return (
            'cirq.DensityMatrixSimulatorState('
            f'density_matrix=np.array({self.density_matrix.tolist()!r}), '
            f'qubit_map={self.qubit_map!r})'
        )


@value.value_equality(unhashable=True)
class DensityMatrixTrialResult(simulator.SimulationTrialResult):
    """A `SimulationTrialResult` for `DensityMatrixSimulator` runs.

    The density matrix that is stored in this result is returned in the
    computational basis with these basis states defined by the qubit_map.
    In particular the value in the qubit_map is the index of the qubit,
    and these are translated into binary vectors where the last qubit is
    the 1s bit of the index, the second-to-last is the 2s bit of the index,
    and so forth (i.e. big endian ordering). The density matrix is a
    `2 ** num_qubits` square matrix, with rows and columns ordered by
    the computational basis as just described.

    Example:
         qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
         Then the returned density matrix will have (row and column) indices
         mapped to qubit basis states like the following table

            |     | QubitA | QubitB | QubitC |
            | :-: | :----: | :----: | :----: |
            |  0  |   0    |   0    |   0    |
            |  1  |   0    |   0    |   1    |
            |  2  |   0    |   1    |   0    |
            |  3  |   0    |   1    |   1    |
            |  4  |   1    |   0    |   0    |
            |  5  |   1    |   0    |   1    |
            |  6  |   1    |   1    |   0    |
            |  7  |   1    |   1    |   1    |

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_simulator_state: The final simulator state of the system after the
            trial finishes.
    """

    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_step_result: DensityMatrixStepResult,
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_step_result=final_step_result
        )
        self._final_density_matrix: Optional[np.ndarray] = None

    @property
    def final_density_matrix(self):
        if self._final_density_matrix is None:
            size = np.prod(protocols.qid_shape(self), dtype=np.int64)
            self._final_density_matrix = np.reshape(
                self._final_simulator_state.density_matrix.copy(), (size, size)
            )
        return self._final_density_matrix

    def _value_equality_values_(self) -> Any:
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return (self.params, measurements, self._final_simulator_state)

    def __str__(self) -> str:
        samples = super().__str__()
        return f'measurements: {samples}\nfinal density matrix:\n{self.final_density_matrix}'

    def __repr__(self) -> str:
        return (
            'cirq.DensityMatrixTrialResult('
            f'params={self.params!r}, measurements={self.measurements!r}, '
            f'final_simulator_state={self._final_simulator_state!r})'
        )

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text("cirq.DensityMatrixTrialResult(...)" if cycle else self.__str__())

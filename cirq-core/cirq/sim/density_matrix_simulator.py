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
from typing import Any, Dict, List, TYPE_CHECKING, Tuple, Union, Sequence

import numpy as np

from cirq import ops, protocols, qis, study, value
from cirq.sim import density_matrix_utils, simulator, act_on_density_matrix_args, simulator_base

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
):
    """A simulator for density matrices and noisy quantum circuits.

    This simulator can be applied on circuits that are made up of operations
    that have:
        * a `_channel_` method
        * a `_mixture_` method for a probabilistic combination of unitary gates.
        * a `_unitary_` method
        * a `_has_unitary_` and `_apply_unitary_` method.
        * measurements
        * a `_decompose_` that eventually yields one of the above
    That is, the circuit must have elements that follow on of the protocols:
        * `cirq.SupportsChannel`
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

               will raise a `ValueError` exception if you call `simulator.run`
               when `ignore_measurement_results` has been set to True
               (for more see https://github.com/quantumlib/Cirq/issues/2777).
        """
        super().__init__(
            dtype=dtype,
            noise=noise,
            seed=seed,
            ignore_measurement_results=ignore_measurement_results,
        )
        if dtype not in {np.complex64, np.complex128}:
            raise ValueError(f'dtype must be complex64 or complex128, was {dtype}')

    def _create_act_on_args(
        self,
        initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE', 'cirq.ActOnDensityMatrixArgs'],
        qubits: Sequence['cirq.Qid'],
    ) -> 'cirq.ActOnDensityMatrixArgs':
        """Creates the ActOnDensityMatrixArgs for a circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

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
            axes=[],
            qid_shape=qid_shape,
            prng=self._prng,
            log_of_measurement_results={},
        )

    def _can_be_in_run_prefix(self, val: Any):
        return not protocols.is_measurement(val)

    def _create_step_result(
        self,
        sim_state: act_on_density_matrix_args.ActOnDensityMatrixArgs,
        qubit_map: Dict['cirq.Qid', int],
    ):
        return DensityMatrixStepResult(
            density_matrix=sim_state.target_tensor,
            measurements=dict(sim_state.log_of_measurement_results),
            qubit_map=qubit_map,
            dtype=self._dtype,
        )

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'DensityMatrixSimulatorState',
    ) -> 'DensityMatrixTrialResult':
        return DensityMatrixTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )


class DensityMatrixStepResult(simulator.StepResult['DensityMatrixSimulatorState']):
    """A single step in the simulation of the DensityMatrixSimulator.

    Attributes:
        qubit_map: A map from the Qubits in the Circuit to the the index
            of this qubit for a canonical ordering. This canonical ordering
            is used to define the state vector (see the state_vector()
            method).
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(
        self,
        density_matrix: np.ndarray,
        measurements: Dict[str, np.ndarray],
        qubit_map: Dict[ops.Qid, int],
        dtype: 'DTypeLike' = np.complex64,
    ):
        """DensityMatrixStepResult.

        Args:
            density_matrix: The density matrix at this step. Can be mutated.
            measurements: The measurements for this step of the simulation.
            qubit_map: A map from qid to index used to define the
                ordering of the basis in density_matrix.
            dtype: The numpy dtype for the density matrix.
        """
        super().__init__(measurements)
        self._density_matrix = density_matrix
        self._qubit_map = qubit_map
        self._dtype = dtype
        self._qid_shape = simulator._qubit_map_to_shape(qubit_map)

    def _qid_shape_(self):
        return self._qid_shape

    def _simulator_state(self) -> 'DensityMatrixSimulatorState':
        return DensityMatrixSimulatorState(self._density_matrix, self._qubit_map)

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
        density_matrix = qis.to_valid_density_matrix(
            density_matrix_repr, len(self._qubit_map), qid_shape=self._qid_shape, dtype=self._dtype
        )
        sim_state_matrix = self._simulator_state().density_matrix
        density_matrix = np.reshape(density_matrix, sim_state_matrix.shape)
        np.copyto(dst=sim_state_matrix, src=density_matrix)

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
        size = np.prod(self._qid_shape, dtype=int)
        matrix = self._density_matrix.copy() if copy else self._density_matrix
        return np.reshape(matrix, (size, size))

    def sample(
        self,
        qubits: List[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        indices = [self._qubit_map[q] for q in qubits]
        return density_matrix_utils.sample_density_matrix(
            self._simulator_state().density_matrix,
            indices,
            qid_shape=self._qid_shape,
            repetitions=repetitions,
            seed=seed,
        )


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
        final_density_matrix: The final density matrix of the system.
    """

    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: DensityMatrixSimulatorState,
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )
        size = np.prod(protocols.qid_shape(self), dtype=int)
        self.final_density_matrix = np.reshape(
            final_simulator_state.density_matrix.copy(), (size, size)
        )

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

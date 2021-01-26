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

import collections
from typing import Any, Dict, Iterator, List, TYPE_CHECKING, Tuple, Type, Union

import numpy as np

from cirq import circuits, ops, protocols, qis, study, value, devices
from cirq.sim import density_matrix_utils, simulator
from cirq.sim.abstract_state import AbstractState
from cirq.sim.op_by_op_simulator import StateFactory, SimulationResultFactory, OpByOpSimulator
from cirq.sim.simulator import check_all_resolved

if TYPE_CHECKING:
    from typing import Tuple
    import cirq


class _StateAndBuffers(AbstractState):
    def __init__(
        self,
        num_qubits: int,
        tensor: np.ndarray,
    ):
        self.num_qubits = num_qubits
        self.tensor = tensor
        self.buffers = [np.empty_like(tensor) for _ in range(3)]
        self._log_of_measurement_results = {}  # type: Dict[str, List[np.ndarray]]

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def log_of_measurement_results(self):
        return self._log_of_measurement_results

    def record_measurement_result(self, key: str, value: Any):
        if key not in self._log_of_measurement_results:
            self._log_of_measurement_results[key] = []
        self._log_of_measurement_results[key].extend(value)


class DensityMatrixStateFactory(StateFactory[_StateAndBuffers]):
    def __init__(
        self,
        *,
        dtype: Type[np.number] = np.complex64,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        ignore_measurement_results: bool = False,
    ):
        if np.dtype(dtype).kind != 'c':
            raise ValueError('dtype must be a complex type but was {}'.format(dtype))
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)
        self._ignore_measurement_results = ignore_measurement_results

    def create_sim_state(self, initial_state, qubits):
        qid_shape = protocols.qid_shape(qubits)
        initial_matrix = qis.to_valid_density_matrix(
            initial_state, len(qid_shape), qid_shape=qid_shape, dtype=self._dtype
        )
        if np.may_share_memory(initial_matrix, initial_state):
            initial_matrix = initial_matrix.copy()

        sim_state = _StateAndBuffers(len(qid_shape), initial_matrix.reshape(qid_shape * 2))
        return sim_state

    def act_on_state(self, op, sim_state: _StateAndBuffers, qubit_map):
        indices = [qubit_map[qubit] for qubit in op.qubits]
        # TODO: support more general measurements.
        # Github issue: https://github.com/quantumlib/Cirq/issues/1357
        if isinstance(op.gate, ops.MeasurementGate):
            meas = op.gate
            if self._ignore_measurement_results:
                for i, q in enumerate(op.qubits):
                    self._apply_op_channel(ops.phase_damp(1).on(q), sim_state, [indices[i]])
            else:
                invert_mask = meas.full_invert_mask()
                # Measure updates inline.
                bits, _ = density_matrix_utils.measure_density_matrix(
                    sim_state.tensor,
                    indices,
                    qid_shape=None,  # sim_state.qid_shape,
                    out=sim_state.tensor,
                    seed=self._prng,
                )
                corrected = [bit ^ (bit < 2 and mask) for bit, mask in zip(bits, invert_mask)]
                key = protocols.measurement_key(meas)
                sim_state.record_measurement_result(key, corrected)
        else:
            self._apply_op_channel(op, sim_state, indices)

    @property
    def prng(self):
        return self._prng

    def on_stuck(self, bad_op: ops.Operation):
        return TypeError(
            "Can't simulate operations that don't implement "
            "SupportsUnitary, SupportsConsistentApplyUnitary, "
            "SupportsMixture, SupportsChannel or is a measurement: {!r}".format(bad_op)
        )

    def keep(self, potential_op: ops.Operation) -> bool:
        return protocols.has_channel(potential_op, allow_decompose=False) or isinstance(
            potential_op.gate, ops.MeasurementGate
        )

    @staticmethod
    def _apply_op_channel(op: ops.Operation, state: _StateAndBuffers, indices: List[int]) -> None:
        """Apply channel to state."""
        result = protocols.apply_channel(
            op,
            args=protocols.ApplyChannelArgs(
                target_tensor=state.tensor,
                out_buffer=state.buffers[0],
                auxiliary_buffer0=state.buffers[1],
                auxiliary_buffer1=state.buffers[2],
                left_axes=indices,
                right_axes=[e + state.num_qubits for e in indices],
            ),
        )
        for i in range(3):
            if result is state.buffers[i]:
                state.buffers[i] = state.tensor
        state.tensor = result


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


class DensityMatrixStepResult(simulator.StepResult):
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
        dtype: Type[np.number] = np.complex64,
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

    def _simulator_state(self) -> DensityMatrixSimulatorState:
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


class DensityMatrixSimulationResultFactory(
    SimulationResultFactory[
        _StateAndBuffers,
        DensityMatrixStepResult,
        DensityMatrixTrialResult,
        DensityMatrixSimulatorState,
    ]
):
    def trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: Any,
    ) -> DensityMatrixTrialResult:
        return DensityMatrixTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def step_result(self, sim_state, qubit_map):
        return DensityMatrixStepResult(
            density_matrix=sim_state.tensor,
            measurements=dict(),  # dict(sim_state.log_of_measurement_results),
            qubit_map=qubit_map,
            dtype=sim_state.dtype,
        )


class DensityMatrixSimulator(OpByOpSimulator[_StateAndBuffers, DensityMatrixStepResult, DensityMatrixTrialResult, DensityMatrixSimulatorState]):
    def __init__(
        self,
        *,
        dtype: Type[np.number] = np.complex64,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        state_algo: DensityMatrixStateFactory = None,
        result_producer: DensityMatrixSimulationResultFactory = None,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        ignore_measurement_results: bool = False,
    ):
        state_algo = DensityMatrixStateFactory(dtype=dtype, seed=seed)
        result_producer = DensityMatrixSimulationResultFactory()
        super().__init__(state_algo, result_producer)


class DensityMatrixSimulatorx(simulator.SimulatesSamples, simulator.SimulatesIntermediateState):
    def __init__(
        self,
        *,
        dtype: Type[np.number] = np.complex64,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        state_algo: DensityMatrixStateFactory = None,
        result_producer: DensityMatrixSimulationResultFactory = None,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        ignore_measurement_results: bool = False,
    ):
        self.state_algo = state_algo or DensityMatrixStateFactory(
            dtype=dtype, seed=seed, ignore_measurement_results=ignore_measurement_results
        )
        self.result_producer = result_producer or DensityMatrixSimulationResultFactory()
        self.noise = devices.NoiseModel.from_noise_model_like(noise)

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)

        if circuit.are_all_measurements_terminal():
            return self._run_sweep_sample(resolved_circuit, repetitions)
        return self._run_sweep_repeat(resolved_circuit, repetitions)

    def _run_sweep_sample(
        self, circuit: circuits.Circuit, repetitions: int
    ) -> Dict[str, np.ndarray]:
        for step_result in self._base_iterator(
            circuit=circuit,
            qubit_order=ops.QubitOrder.DEFAULT,
            initial_state=0,
            all_measurements_are_terminal=True,
        ):
            pass
        measurement_ops = [
            op for _, op, _ in circuit.findall_operations_with_gate_type(ops.MeasurementGate)
        ]
        return step_result.sample_measurement_ops(
            measurement_ops, repetitions, seed=self.state_algo.prng
        )

    def _run_sweep_repeat(
        self, circuit: circuits.Circuit, repetitions: int
    ) -> Dict[str, np.ndarray]:
        measurements = {}  # type: Dict[str, List[np.ndarray]]
        if repetitions == 0:
            for _, op, _ in circuit.findall_operations_with_gate_type(ops.MeasurementGate):
                measurements[protocols.measurement_key(op)] = np.empty([0, 1])

        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                circuit, qubit_order=ops.QubitOrder.DEFAULT, initial_state=0
            )
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=np.uint8))
        return {k: np.array(v) for k, v in measurements.items()}

    def _base_iterator(
        self,
        circuit: circuits.Circuit,
        qubit_order: ops.QubitOrderOrList,
        initial_state: 'cirq.STATE_VECTOR_LIKE',
        perform_measurements: bool = True,
        all_measurements_are_terminal: bool = False,
    ) -> Iterator['DensityMatrixStepResult']:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        sim_state = self.state_algo.create_sim_state(initial_state, qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        if len(circuit) == 0:
            yield self.result_producer.step_result(sim_state, qubit_map)
        measured = collections.defaultdict(bool)  # type: Dict[Tuple[cirq.Qid, ...], bool]

        noisy_moments = self.noise.noisy_moments(circuit, sorted(circuit.all_qubits()))
        for moment in noisy_moments:
            operations = moment
            if self.state_algo.keep:
                operations = protocols.decompose(
                    moment, keep=self.state_algo.keep, on_stuck_raise=self.state_algo.on_stuck
                )
            for op in operations:
                if all_measurements_are_terminal and measured[op.qubits]:
                    continue
                if isinstance(op.gate, ops.MeasurementGate):
                    measured[op.qubits] = True
                    if all_measurements_are_terminal:
                        continue
                if perform_measurements or not isinstance(op.gate, ops.MeasurementGate):
                    self.state_algo.act_on_state(op, sim_state, qubit_map)

            yield self.result_producer.step_result(sim_state, qubit_map)
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

from typing import cast, Dict, Iterator, List, Type, Union

import numpy as np

from cirq import circuits, linalg, ops, protocols, study
from cirq.sim import density_matrix, simulator

class DensityMatrixSimulator(simulator.SimulatesSamples,
                             simulator.SimulatesIntermediateState):

    def __init__(self, dtype: Type[np.number] = np.complex64):
        if dtype not in {np.complex64, np.complex128}:
            raise ValueError(
                'dtype must be complex64 or complex128, was {}'.format(dtype))

        self._dtype = dtype

    def _run(self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, np.ndarray]:
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        measurements = {}  # type: Dict[str, List[np.ndarray]]
        # TODO: optimize for all terminal measurements
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                resolved_circuit,
                qubit_order=ops.QubitOrder.DEFAULT,
                initial_state=0,
                perform_measurements=True)
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=bool))
        return {k: np.array(v) for k, v in measurements.items()}


    def _simulator_iterator(self,
            circuit: circuits.Circuit,
            param_resolver: study.ParamResolver,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray]) -> Iterator:
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        actual_initial_state = 0 if initial_state is None else initial_state
        return self._base_iterator(resolved_circuit,
                                   qubit_order,
                                   actual_initial_state)

    def _base_iterator(
        self,
        circuit: circuits.Circuit,
        qubit_order: ops.QubitOrderOrList,
        initial_state: Union[int, np.ndarray],
        perform_measurements: bool=True,
    ) -> Iterator:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        matrix = density_matrix.to_valid_density_matrix(
            initial_state, num_qubits, self._dtype)
        if len(circuit) == 0:
            yield DensityMatrixStepResult(matrix, {}, qubit_map, self._dtype)

        def on_stuck(bad_op: ops.Operation):
            return TypeError(
                "Can't simulate unknown operations that don't specify a "
                "_unitary_ method, a _decompose_ method, or "
                "(_has_unitary_ + _apply_unitary_) methods"
                ": {!r}".format(bad_op))

        def keep(potential_op: ops.Operation) -> bool:
            return (protocols.has_channel(potential_op) or
                    ops.MeasurementGate.is_measurement(potential_op))

        matrix = np.reshape(matrix, (2,) * num_qubits * 2)
        buffer = np.empty((2,) * 2 * num_qubits, dtype=self._dtype)
        for moment in circuit:
            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[bool]]

            non_display_ops = (op for op in moment
                               if not isinstance(op, (ops.SamplesDisplay,
                                                      ops.WaveFunctionDisplay)))

            channel_ops_and_measurements = protocols.decompose(
                non_display_ops,
                keep=keep,
                on_stuck_raise=on_stuck)

            for op in channel_ops_and_measurements:
                indices = [qubit_map[qubit] for qubit in op.qubits]
                if ops.MeasurementGate.is_measurement(op):
                    gate = cast(ops.MeasurementGate,
                                cast(ops.GateOperation, op).gate)
                    if perform_measurements:
                        invert_mask = gate.invert_mask or num_qubits * (False,)
                        # Measure updates inline.
                        bits, _ = density_matrix.measure_density_matrix(
                            matrix, indices, matrix)
                        corrected = [bit ^ mask for bit, mask in
                                     zip(bits, invert_mask)]
                        measurements[cast(str, gate.key)].extend(corrected)
                else:
                    # TODO: Use apply_channel similar to apply_unitary.
                    gate = cast(ops.GateOperation, op).gate
                    channel = protocols.channel(gate)
                    sum_buffer = np.zeros((2,) * 2 * num_qubits,
                                          dtype=self._dtype)
                    for krauss in channel:
                        krauss_tensor = np.reshape(
                            np.kron(krauss, np.transpose(np.conj(krauss))),
                            (2,) * gate.num_qubits() * 4,).astype(self._dtype)
                        linalg.targeted_left_multiply(krauss_tensor, matrix,
                                                      indices + [i + num_qubits
                                                                 for i in
                                                                 indices],
                                                      buffer)
                        sum_buffer += buffer
                    np.copyto(matrix, sum_buffer)
            yield DensityMatrixStepResult(
                matrix=matrix,
                measurements=measurements,
                qubit_map=qubit_map,
                dtype=self._dtype)


class DensityMatrixStepResult(simulator.StepResult):

    def __init__(self,
        matrix: np.ndarray,
        measurements: Dict[str, np.ndarray],
        qubit_map: Dict[ops.QubitId, int],
        dtype: Type[np.number] = np.complex64):
        super().__init__(measurements)
        self._matrix = matrix
        self._qubit_map = qubit_map
        self.dtype = dtype


    def simulator_state(self) -> 'DensityMatrixSimulatorState':
        return DensityMatrixSimulatorState(self._matrix, self._qubit_map)

    def set_density_matrix(self, density_matrix_repr: Union[int, np.ndarray]):
        self._matrix = density_matrix.to_valid_density_matrix(
            density_matrix, len(self._qubit_map), self._dtype)

    def sample(self, qubits: List[ops.QubitId],
        repetitions: int = 1) -> np.ndarray:
        indices = [self._qubit_map[q] for q in qubits]
        density_matrix.sample_density_matrix(self.simulator_state().matrix,
                                             indices, repetitions)

class DensityMatrixSimulatorState():

    def __init__(self,
        matrix: np.ndarray,
        qubit_map: Dict[ops.QubitId, int]):
        self.matrix = matrix
        self.qubit_map = qubit_map



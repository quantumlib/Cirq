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
"""An MPS simulator.

https://arxiv.org/abs/2002.07730
"""

import collections
from typing import Any, Dict, List, Iterator, Sequence

import numpy as np
from cirq.ops.global_phase_op import GlobalPhaseOperation

import cirq
from cirq import circuits, study, ops, protocols, value
from cirq.ops.dense_pauli_string import DensePauliString
from cirq.protocols import act_on, unitary
from cirq.sim import simulator


class MPSSimulator(simulator.SimulatesSamples, simulator.SimulatesIntermediateState):
    """An efficient simulator for MPS circuits."""

    def __init__(self, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        """Creates instance of `MPSSimulator`.

        Args:
            seed: The random seed to use for this simulator.
        """
        self.init = True
        self._prng = value.parse_random_state(seed)

    @staticmethod
    def is_supported_operation(op: 'cirq.Operation') -> bool:
        """Checks whether given operation can be simulated by this simulator."""
        if protocols.has_unitary(op):
            return True
        else:
            return op.gate in [cirq.CNOT, cirq.CZ]

    def _base_iterator(
        self, circuit: circuits.Circuit, qubit_order: ops.QubitOrderOrList, initial_state: int
    ) -> Iterator['cirq.MPSSimulatorStepResult']:
        """Iterator over MPSSimulatorStepResult from Moments of a Circuit

        Args:
            circuit: The circuit to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.


        Yields:
            MPSStepResult from simulating a Moment of the Circuit.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())

        qubit_map = {q: i for i, q in enumerate(qubits)}

        if len(circuit) == 0:
            yield MPSSimulatorStepResult(
                measurements={}, state=MPSState(qubit_map, initial_state=initial_state)
            )
            return

        state = MPSState(qubit_map, initial_state=initial_state)

        for moment in circuit:
            measurements: Dict[str, List[np.ndarray]] = collections.defaultdict(list)

            for op in moment:
                if isinstance(op.gate, ops.MeasurementGate):
                    key = protocols.measurement_key(op)
                    measurements[key].extend(state.perform_measurement(op.qubits, self._prng))
                elif protocols.has_unitary(op):
                    state.apply_unitary(op)
                else:
                    raise NotImplementedError(f"Unrecognized operation: {op!r}")

            yield MPSSimulatorStepResult(measurements=measurements, state=state)

    def _simulator_iterator(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        qubit_order: ops.QubitOrderOrList,
        initial_state: int,
    ) -> Iterator:
        """See definition in `cirq.SimulatesIntermediateState`.

        Args:
            inital_state: An integer specifying the inital
            state in the computational basis.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        self._check_all_resolved(resolved_circuit)
        actual_initial_state = 0 if initial_state is None else initial_state

        return self._base_iterator(resolved_circuit, qubit_order, actual_initial_state)

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state,
    ):

        return MPSTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, List[np.ndarray]]:

        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        self._check_all_resolved(resolved_circuit)

        measurements = {}  # type: Dict[str, List[np.ndarray]]
        if repetitions == 0:
            for _, op, _ in resolved_circuit.findall_operations_with_gate_type(ops.MeasurementGate):
                measurements[protocols.measurement_key(op)] = np.empty([0, 1])

        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                resolved_circuit, qubit_order=ops.QubitOrder.DEFAULT, initial_state=0
            )

            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=bool))

        return {k: np.array(v) for k, v in measurements.items()}

    def _check_all_resolved(self, circuit):
        """Raises if the circuit contains unresolved symbols."""
        if protocols.is_parameterized(circuit):
            unresolved = [
                op for moment in circuit for op in moment if protocols.is_parameterized(op)
            ]
            raise ValueError(
                'Circuit contains ops whose symbols were not specified in '
                'parameter sweep. Ops: {}'.format(unresolved)
            )


class MPSTrialResult(simulator.SimulationTrialResult):
    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'MPSState',
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

        self.final_state = final_simulator_state

    def __str__(self) -> str:
        samples = super().__str__()
        final = self._final_simulator_state
        return f'measurements: {samples}\noutput state: {final}'


class MPSSimulatorStepResult(simulator.StepResult):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, state, measurements):
        """Results of a step of the simulator.
        Attributes:
            state: A MPSState
            measurements: A dictionary from measurement gate key to measurement
                results, ordered by the qubits that the measurement operates on.
            qubit_map: A map from the Qubits in the Circuit to the the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state vector (see the state_vector()
                method).
        """
        self.measurements = measurements
        self.state = state

    def __str__(self) -> str:
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        results = sorted([(key, bitstring(val)) for key, val in self.measurements.items()])

        if len(results) == 0:
            measurements = ''
        else:
            measurements = ' '.join([f'{key}={val}' for key, val in results]) + '\n'

        final = self.state

        return f'{measurements}{final}'

    def _simulator_state(self):
        return self.state

    def sample(
        self,
        qubits: List[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:

        measurements = []

        for _ in range(repetitions):
            measurements.append(
                self.state.perform_measurement(
                    qubits, value.parse_random_state(seed), collapse_state_vector=False
                )
            )

        return np.array(measurements, dtype=bool)


@value.value_equality
class MPSState:
    """A state of the MPS simulation."""

    def __init__(self, qubit_map, initial_state=0):
        self.qubit_map = qubit_map
        self.M = []
        for qubit in qubit_map.keys():
            d = qubit.dimension
            x = np.zeros((1, 1, d,))
            x[0, 0, (initial_state % d)] = 1.0
            self.M.append(x)
            initial_state = initial_state // d

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__,
            'qubit_map': [(k, v) for k, v in self.qubit_map.items()],
            'M': self.M,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_map, **kwargs):
        state = cls(dict(qubit_map))
        return state

    def _value_equality_values_(self) -> Any:
        return self.qubit_map

    def copy(self) -> 'MPSState':
        state = MPSState(self.qubit_map)
        return state

    def state_vector(self):
        return np.asarray([0.0])

    def to_numpy(self) -> np.ndarray:
        return self.state_vector()

    def apply_unitary(self, op: 'cirq.Operation'):
        U = protocols.unitary(op)
        print('TONYBOOM apply_unitary() op=%s U=%s' % (op, U))
        return

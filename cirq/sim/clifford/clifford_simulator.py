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
"""An efficient simulator for Clifford circuits.

Allowed operations include:
	- X,Y,Z,H,S,CNOT,CZ
	- measurements in the computational basis

The quantum state is specified in two forms:
    1. In terms of stabilizer generators. These are a set of n Pauli operators
    {S_1,S_2,...,S_n} such that S_i |psi> = |psi>.

    This implementation is based on Aaronson and Gottesman,
    2004 (arXiv:quant-ph/0406196).

    2. In the CH-form defined by Bravyi et al, 2018 (arXiv:1808.00128).
    This representation keeps track of overall phase and enables access
    to state vector amplitudes.
"""

import collections
from typing import Any, Dict, List, Iterator, Sequence

import numpy as np
from cirq.ops.global_phase_op import GlobalPhaseOperation

import cirq
from cirq import circuits, study, ops, protocols, value
from cirq.ops import pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.ops.dense_pauli_string import DensePauliString
from cirq.protocols import unitary
from cirq.sim import simulator
from cirq.sim.clifford import clifford_tableau, stabilizer_state_ch_form
from cirq._compat import deprecated, deprecated_parameter


class CliffordSimulator(simulator.SimulatesSamples,
                        simulator.SimulatesIntermediateState):
    """An efficient simulator for Clifford circuits."""

    def __init__(self, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        """Creates instance of `CliffordSimulator`.

        Args:
            seed: The random seed to use for this simulator.
        """
        self.init = True
        self._prng = value.parse_random_state(seed)

    @staticmethod
    def is_supported_operation(op: 'cirq.Operation') -> bool:
        """Checks whether given operation can be simulated by this simulator."""
        # TODO: support more general Pauli measurements
        if isinstance(op.gate, cirq.MeasurementGate): return True
        if isinstance(op, GlobalPhaseOperation): return True
        if not protocols.has_unitary(op): return False
        if len(op.qubits) == 1:
            u = unitary(op)
            return SingleQubitCliffordGate.from_unitary(u) is not None
        else:
            return op.gate in [cirq.CNOT, cirq.CZ]

    def _base_iterator(self, circuit: circuits.Circuit,
                       qubit_order: ops.QubitOrderOrList, initial_state: int
                      ) -> Iterator['cirq.CliffordSimulatorStepResult']:
        """Iterator over CliffordSimulatorStepResult from Moments of a Circuit

        Args:
            circuit: The circuit to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation.


        Yields:
            CliffordStepResult from simulating a Moment of the Circuit.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())

        qubit_map = {q: i for i, q in enumerate(qubits)}

        if len(circuit) == 0:
            yield CliffordSimulatorStepResult(measurements={},
                                              state=CliffordState(
                                                  qubit_map,
                                                  initial_state=initial_state))
            return

        state = CliffordState(qubit_map, initial_state=initial_state)

        for moment in circuit:
            measurements: Dict[str, List[np.ndarray]] = collections.defaultdict(
                list)

            for op in moment:
                if isinstance(op.gate, ops.MeasurementGate):
                    key = protocols.measurement_key(op)
                    measurements[key].extend(
                        state.perform_measurement(op.qubits, self._prng))
                elif protocols.has_unitary(op):
                    state.apply_unitary(op)
                else:
                    raise NotImplementedError(f"Unrecognized operation: {op!r}")

            yield CliffordSimulatorStepResult(measurements=measurements,
                                              state=state)

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

        return self._base_iterator(resolved_circuit, qubit_order,
                                   actual_initial_state)

    def _create_simulator_trial_result(self, params: study.ParamResolver,
                                       measurements: Dict[str, np.ndarray],
                                       final_simulator_state):

        return CliffordTrialResult(params=params,
                                   measurements=measurements,
                                   final_simulator_state=final_simulator_state)

    def _run(self, circuit: circuits.Circuit,
             param_resolver: study.ParamResolver,
             repetitions: int) -> Dict[str, List[np.ndarray]]:

        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        self._check_all_resolved(resolved_circuit)

        measurements = {}  # type: Dict[str, List[np.ndarray]]
        if repetitions == 0:
            for _, op, _ in resolved_circuit.findall_operations_with_gate_type(
                    ops.MeasurementGate):
                measurements[protocols.measurement_key(op)] = np.empty([0, 1])

        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                resolved_circuit,
                qubit_order=ops.QubitOrder.DEFAULT,
                initial_state=0)

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
                op for moment in circuit for op in moment
                if protocols.is_parameterized(op)
            ]
            raise ValueError(
                'Circuit contains ops whose symbols were not specified in '
                'parameter sweep. Ops: {}'.format(unresolved))


class CliffordTrialResult(simulator.SimulationTrialResult):

    def __init__(self, params: study.ParamResolver,
                 measurements: Dict[str, np.ndarray],
                 final_simulator_state: 'CliffordState') -> None:
        super().__init__(params=params,
                         measurements=measurements,
                         final_simulator_state=final_simulator_state)

        self.final_state = final_simulator_state

    def __str__(self) -> str:
        samples = super().__str__()
        final = self._final_simulator_state
        return f'measurements: {samples}\noutput state: {final}'


class CliffordSimulatorStepResult(simulator.StepResult):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, state, measurements):
        """Results of a step of the simulator.
        Attributes:
            state: A CliffordState
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

        results = sorted([
            (key, bitstring(val)) for key, val in self.measurements.items()
        ])

        if len(results) == 0:
            measurements = ''
        else:
            measurements = ' '.join([f'{key}={val}' for key, val in results
                                    ]) + '\n'

        final = self.state

        return f'{measurements}{final}'

    def _simulator_state(self):
        return self.state

    def sample(self,
               qubits: List[ops.Qid],
               repetitions: int = 1,
               seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None) -> np.ndarray:

        measurements = []

        for _ in range(repetitions):
            measurements.append(
                self.state.perform_measurement(qubits,
                                               value.parse_random_state(seed),
                                               collapse_state_vector=False))

        return np.array(measurements, dtype=bool)


@value.value_equality
class CliffordState():
    """A state of the Clifford simulation.

    The state is stored using two complementary representations:
    Anderson's tableaux form and Bravyi's CH-form.
    The tableaux keeps track of the stabilizer operations, while the
    CH-form allows access to the full state vector (including phase).

    Gates and measurements are applied to each representation in O(n^2) time.
    """

    def __init__(self, qubit_map, initial_state=0):
        self.qubit_map = qubit_map
        self.n = len(qubit_map)

        self.tableau = clifford_tableau.CliffordTableau(self.n, initial_state)
        self.ch_form = stabilizer_state_ch_form.StabilizerStateChForm(
            self.n, initial_state)

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__,
            'qubit_map': [(k, v) for k, v in self.qubit_map.items()],
            'tableau': self.tableau,
            'ch_form': self.ch_form,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_map, tableau, ch_form, **kwargs):
        state = cls(dict(qubit_map))
        state.tableau = tableau
        state.ch_form = ch_form

        return state

    def _value_equality_values_(self) -> Any:
        return self.qubit_map, self.tableau, self.ch_form

    def copy(self) -> 'CliffordState':
        state = CliffordState(self.qubit_map)
        state.tableau = self.tableau.copy()
        state.ch_form = self.ch_form.copy()

        return state

    def __repr__(self) -> str:
        return repr(self.ch_form)

    def __str__(self) -> str:
        """Return the state vector string representation of the state."""
        return str(self.ch_form)

    def to_numpy(self) -> np.ndarray:
        return self.ch_form.to_state_vector()

    def stabilizers(self) -> List[DensePauliString]:
        """Returns the stabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that S_i |psi> = |psi> """
        return self.tableau.stabilizers()

    def destabilizers(self) -> List[DensePauliString]:
        """Returns the destabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that along with the stabilizer
        generators above generate the full Pauli group on n qubits."""
        return self.tableau.destabilizers()

    def state_vector(self):
        return self.ch_form.state_vector()

    @deprecated(deadline='v0.10.0', fix='use state_vector instead')
    def wave_function(self):
        return self.state_vector()

    def apply_unitary(self, op: 'cirq.Operation'):
        if len(op.qubits) == 1:
            self.apply_single_qubit_unitary(op)
        elif isinstance(op, GlobalPhaseOperation):
            self.ch_form.omega *= op.coefficient
        elif op.gate == cirq.CNOT:
            self.tableau._CNOT(self.qubit_map[op.qubits[0]],
                               self.qubit_map[op.qubits[1]])
            self.ch_form._CNOT(self.qubit_map[op.qubits[0]],
                               self.qubit_map[op.qubits[1]])
        elif op.gate == cirq.CZ:
            self.tableau._CZ(self.qubit_map[op.qubits[0]],
                             self.qubit_map[op.qubits[1]])
            self.ch_form._CZ(self.qubit_map[op.qubits[0]],
                             self.qubit_map[op.qubits[1]])
        else:
            raise ValueError('%s cannot be run with Clifford simulator.' %
                             str(op.gate))  # type: ignore

    def apply_single_qubit_unitary(self, op: 'cirq.Operation'):
        qubit = self.qubit_map[op.qubits[0]]
        if op.gate == cirq.I:
            return

        if op.gate == cirq.X:
            self._apply_X(qubit)
            return

        if op.gate == cirq.Y:
            self._apply_Y(qubit)
            return

        if op.gate == cirq.Z:
            self._apply_Z(qubit)
            return

        if op.gate == cirq.H:
            self._apply_H(qubit)
            return

        u = unitary(op)
        clifford_gate = SingleQubitCliffordGate.from_unitary(u)
        if clifford_gate is None:
            raise ValueError('%s cannot be run with Clifford simulator.' %
                             str(op.gate))

        h = unitary(ops.H)
        s = unitary(ops.S)
        applied_unitary = np.eye(2)
        for axis, quarter_turns in clifford_gate.decompose_rotation():
            for _ in range(quarter_turns % 4):
                if axis == pauli_gates.X:
                    self._apply_H(qubit)
                    self._apply_S(qubit)
                    self._apply_H(qubit)
                    applied_unitary = h @ s @ h @ applied_unitary
                elif axis == pauli_gates.Y:
                    self._apply_S(qubit)
                    self._apply_S(qubit)
                    self._apply_H(qubit)
                    applied_unitary = h @ s @ s @ applied_unitary
                else:
                    assert axis == pauli_gates.Z
                    self._apply_S(qubit)
                    applied_unitary = s @ applied_unitary

        max_idx = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
        phase_shift = u[max_idx] / applied_unitary[max_idx]
        self.ch_form.omega *= phase_shift

    def _apply_H(self, qubit: int):
        self.tableau._H(qubit)
        self.ch_form._H(qubit)

    def _apply_S(self, qubit: int):
        self.tableau._S(qubit)
        self.ch_form._S(qubit)

    def _apply_X(self, qubit: int):
        self.tableau._X(qubit)
        self.ch_form._X(qubit)

    def _apply_Z(self, qubit: int):
        self.tableau._Z(qubit)
        self.ch_form._Z(qubit)

    def _apply_Y(self, qubit: int):
        self.tableau._Y(qubit)
        self.ch_form._Y(qubit)

    @deprecated_parameter(
        deadline='v0.10.0',
        fix='Use collapse_state_vector instead.',
        parameter_desc='collapse_wavefunction',
        match=lambda args, kwargs: 'collapse_wavefunction' in kwargs,
        rewrite=lambda args, kwargs: (args, {('collapse_state_vector' if k ==
                                              'collapse_wavefunction' else k): v
                                             for k, v in kwargs.items()}))
    def perform_measurement(self,
                            qubits: Sequence[ops.Qid],
                            prng: np.random.RandomState,
                            collapse_state_vector=True):
        results = []

        if collapse_state_vector:
            state = self
        else:
            state = self.copy()

        for qubit in qubits:
            result = state.tableau._measure(self.qubit_map[qubit], prng)
            state.ch_form.project_Z(self.qubit_map[qubit], result)
            results.append(result)

        return results

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

from typing import Any, Dict, List, Iterator, Sequence

import numpy as np

import cirq
from cirq import circuits, study, ops, protocols, value
from cirq.ops.dense_pauli_string import DensePauliString
from cirq.protocols import act_on
from cirq.sim import clifford, simulator
from cirq._compat import deprecated, deprecated_parameter
from cirq.sim.simulator import check_all_resolved


class CliffordSimulator(simulator.SimulatesSamples, simulator.SimulatesIntermediateState):
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
        return protocols.has_stabilizer_effect(op)

    def _base_iterator(
        self, circuit: circuits.Circuit, qubit_order: ops.QubitOrderOrList, initial_state: int
    ) -> Iterator['cirq.CliffordSimulatorStepResult']:
        """Iterator over CliffordSimulatorStepResult from Moments of a Circuit

        Args:
            circuit: The circuit to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.


        Yields:
            CliffordStepResult from simulating a Moment of the Circuit.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())

        qubit_map = {q: i for i, q in enumerate(qubits)}

        if len(circuit) == 0:
            yield CliffordSimulatorStepResult(
                measurements={}, state=CliffordState(qubit_map, initial_state=initial_state)
            )
            return

        state = CliffordState(qubit_map, initial_state=initial_state)
        ch_form_args = clifford.ActOnStabilizerCHFormArgs(
            state.ch_form,
            [],
            self._prng,
            {},
        )

        for moment in circuit:
            ch_form_args.log_of_measurement_results = {}

            for op in moment:
                try:
                    ch_form_args.axes = tuple(state.qubit_map[i] for i in op.qubits)
                    act_on(op, ch_form_args)
                except TypeError:
                    raise NotImplementedError(
                        f"CliffordSimulator doesn't support {op!r}"
                    )  # type: ignore

            yield CliffordSimulatorStepResult(
                measurements=ch_form_args.log_of_measurement_results, state=state
            )

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state,
    ):

        return CliffordTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, List[np.ndarray]]:

        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)

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


class CliffordTrialResult(simulator.SimulationTrialResult):
    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'CliffordState',
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

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
        self.state = state.copy()

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

        measurements = {}  # type: Dict[str, List[np.ndarray]]

        for i in range(repetitions):
            self.state.apply_measurement(
                cirq.measure(*qubits, key=str(i)),
                measurements,
                value.parse_random_state(seed),
                collapse_state_vector=False,
            )

        return np.array(list(measurements.values()), dtype=bool)


@value.value_equality
class CliffordState:
    """A state of the Clifford simulation.

    The state is stored using Bravyi's CH-form which allows access to the full
    state vector (including phase).

    Gates and measurements are applied to each representation in O(n^2) time.
    """

    def __init__(self, qubit_map, initial_state=0):
        self.qubit_map = qubit_map
        self.n = len(qubit_map)

        self.ch_form = clifford.StabilizerStateChForm(self.n, initial_state)

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__,
            'qubit_map': [(k, v) for k, v in self.qubit_map.items()],
            'ch_form': self.ch_form,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_map, ch_form, **kwargs):
        state = cls(dict(qubit_map))
        state.ch_form = ch_form

        return state

    def _value_equality_values_(self) -> Any:
        return self.qubit_map, self.ch_form

    def copy(self) -> 'CliffordState':
        state = CliffordState(self.qubit_map)
        state.ch_form = self.ch_form.copy()

        return state

    def __repr__(self) -> str:
        return repr(self.ch_form)

    def __str__(self) -> str:
        """Return the state vector string representation of the state."""
        return str(self.ch_form)

    def to_numpy(self) -> np.ndarray:
        return self.ch_form.to_state_vector()

    @deprecated(deadline='v0.11.0', fix='use CliffordTableau instead')
    def stabilizers(self) -> List[DensePauliString]:
        """Returns the stabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that S_i |psi> = |psi>"""
        return []

    @deprecated(deadline='v0.11.0', fix='use CliffordTableau instead')
    def destabilizers(self) -> List[DensePauliString]:
        """Returns the destabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that along with the stabilizer
        generators above generate the full Pauli group on n qubits."""
        return []

    def state_vector(self):
        return self.ch_form.state_vector()

    @deprecated(deadline='v0.10.0', fix='use state_vector instead')
    def wave_function(self):
        return self.state_vector()

    def apply_unitary(self, op: 'cirq.Operation'):
        ch_form_args = clifford.ActOnStabilizerCHFormArgs(
            self.ch_form, [self.qubit_map[i] for i in op.qubits], np.random.RandomState(), {}
        )
        try:
            act_on(op, ch_form_args)
        except TypeError:
            raise ValueError(
                '%s cannot be run with Clifford simulator.' % str(op.gate)
            )  # type: ignore
        return

    def apply_measurement(
        self,
        op: 'cirq.Operation',
        measurements: Dict[str, List[np.ndarray]],
        prng: np.random.RandomState,
        collapse_state_vector=True,
    ):
        if not isinstance(op.gate, cirq.MeasurementGate):
            raise TypeError(
                'apply_measurement only supports cirq.MeasurementGate operations. Found %s instead.'
                % str(op.gate)
            )

        if collapse_state_vector:
            state = self
        else:
            state = self.copy()

        qids = [self.qubit_map[i] for i in op.qubits]

        ch_form_args = clifford.ActOnStabilizerCHFormArgs(state.ch_form, qids, prng, measurements)
        act_on(op, ch_form_args)

    @deprecated_parameter(
        deadline='v0.10.0',
        fix='Use collapse_state_vector instead.',
        parameter_desc='collapse_wavefunction',
        match=lambda args, kwargs: 'collapse_wavefunction' in kwargs,
        rewrite=lambda args, kwargs: (
            args,
            {
                ('collapse_state_vector' if k == 'collapse_wavefunction' else k): v
                for k, v in kwargs.items()
            },
        ),
    )
    @deprecated(deadline='v0.11.0', fix='Use the apply_measurement instead')
    def perform_measurement(
        self, qubits: Sequence[ops.Qid], prng: np.random.RandomState, collapse_state_vector=True
    ):
        results = []

        if collapse_state_vector:
            state = self
        else:
            state = self.copy()

        for qubit in qubits:
            result = state.ch_form._measure(self.qubit_map[qubit], prng)
            results.append(result)

        return results

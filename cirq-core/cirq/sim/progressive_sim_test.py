# Copyright 2021 The Cirq Developers
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
import math
from typing import List, Dict, Any, Sequence, Union

import numpy as np

import cirq


def int_to_bool_list(num, bits):
    return [bool(num & (1 << n)) for n in range(bits)]


class PureActOnArgs(cirq.ActOnArgs):
    b = False

    def __init__(self, b: List[bool], qubits, logs):
        super().__init__(
            qubits=qubits,
            log_of_measurement_results=logs,
        )
        self.b = b

    def as_int(self):
        return sum(1 << i for i, v in enumerate(reversed(self.b)) if v)

    def state_vector(self):
        return cirq.to_valid_state_vector(self.as_int(), len(self.qubits))

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        return [1 if self.b[i] else 0 for i in self.get_axes(qubits)]

    def _on_copy(self, target: 'PureActOnArgs'):
        target.b = self.b.copy()

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        gate = action if isinstance(action, cirq.Gate) else action.gate
        qubits = action.qubits if isinstance(action, cirq.Operation) else qubits
        if isinstance(gate, cirq.XPowGate):
            for i in self.get_axes(qubits):
                self.b[i] = not self.b[i]

            return True
        elif isinstance(gate, cirq.ResetChannel):
            for i in self.get_axes(qubits):
                self.b[i] = False
            return True
        return False

    def sample(self, qubits, repetitions=1, seed=None):
        measurements: List[List[int]] = []

        for _ in range(repetitions):
            measurements.append(self._perform_measurement(qubits))

        return np.array(measurements, dtype=int)

    def _on_kronecker_product(self, other: 'PureActOnArgs', target: 'PureActOnArgs'):
        target.b = self.b + other.b


class ProgressiveActOnArgs(cirq.ActOnArgs):
    args: cirq.ActOnArgs = None

    def __init__(self, args, qubits, logs):
        super().__init__(
            qubits=qubits,
            log_of_measurement_results=logs,
        )
        self.args = args

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        return self.args._perform_measurement(qubits)

    def _on_copy(self, target: 'ProgressiveActOnArgs'):
        target.args = self.args.copy()
        target.args._log_of_measurement_results = self._log_of_measurement_results

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        if isinstance(action, cirq.Operation):
            qubits = None
        if isinstance(self.args, PureActOnArgs):
            if self.args._act_on_fallback_(action, qubits, allow_decompose=allow_decompose):
                return True
            ch = cirq.StabilizerStateChForm(len(self.args.qubits), self.args.as_int())
            self.args = cirq.ActOnStabilizerCHFormArgs(
                ch, self.prng, self.log_of_measurement_results, self.qubits
            )
        if isinstance(self.args, cirq.ActOnStabilizerCHFormArgs):
            if cirq.has_stabilizer_effect(action):
                cirq.act_on(action, self.args, qubits, allow_decompose=allow_decompose)
                return True
            sv = self.args.state.state_vector()
            self.args = cirq.ActOnStateVectorArgs(
                sv, np.empty_like(sv), self.prng, self.log_of_measurement_results, self.qubits
            )
        if isinstance(self.args, cirq.ActOnStateVectorArgs):
            if cirq.has_unitary(action):
                cirq.act_on(action, self.args, qubits, allow_decompose=allow_decompose)
                return True
            dm = cirq.density_matrix_from_state_vector(self.args.target_tensor)
            self.args = cirq.ActOnDensityMatrixArgs(
                dm,
                [np.empty_like(dm) for _ in range(3)],
                dm.shape,
                self.prng,
                self.log_of_measurement_results,
                self.qubits,
            )
        if isinstance(self.args, cirq.DensityMatrixSimulator()):
            cirq.act_on(action, self.args, qubits, allow_decompose=allow_decompose)
            return True
        return False

    def sample(self, qubits, repetitions=1, seed=None):
        return self.args.sample(qubits, repetitions, seed)

    def _on_kronecker_product(self, other: 'ProgressiveActOnArgs', target: 'ProgressiveActOnArgs'):
        self_args = self.args
        other_args = other.args
        if isinstance(self_args, PureActOnArgs) and not isinstance(other_args, PureActOnArgs):
            ch = cirq.StabilizerStateChForm(len(self_args.qubits), self_args.as_int())
            self_args = cirq.ActOnStabilizerCHFormArgs(
                ch, self.prng, self.log_of_measurement_results, self.qubits
            )
        if not isinstance(self_args, PureActOnArgs) and isinstance(other_args, PureActOnArgs):
            ch = cirq.StabilizerStateChForm(len(other_args.qubits), other_args.as_int())
            other_args = cirq.ActOnStabilizerCHFormArgs(
                ch, other.prng, other.log_of_measurement_results, other.qubits, other.axes
            )
        if isinstance(self_args, cirq.ActOnStabilizerCHFormArgs) and not isinstance(
            other_args, cirq.ActOnStabilizerCHFormArgs
        ):
            sv = self_args.state.state_vector()
            self_args = cirq.ActOnStateVectorArgs(
                sv, np.empty_like(sv), self.prng, self.log_of_measurement_results, self.qubits
            )
        if not isinstance(self_args, cirq.ActOnStabilizerCHFormArgs) and isinstance(
            other_args, cirq.ActOnStabilizerCHFormArgs
        ):
            sv = other_args.state.state_vector()
            other_args = cirq.ActOnStateVectorArgs(
                sv,
                np.empty_like(sv),
                other.prng,
                other.log_of_measurement_results,
                other.qubits,
                other.axes,
            )
        if isinstance(self_args, cirq.ActOnStateVectorArgs) and not isinstance(
            other_args, cirq.ActOnStateVectorArgs
        ):
            dm = cirq.density_matrix_from_state_vector(self_args.target_tensor)
            self_args = cirq.ActOnDensityMatrixArgs(
                dm,
                [np.empty_like(dm) for _ in range(3)],
                dm.shape,
                self.prng,
                self.log_of_measurement_results,
                self.qubits,
            )
        if not isinstance(self_args, cirq.ActOnStateVectorArgs) and isinstance(
            other_args, cirq.ActOnStateVectorArgs
        ):
            dm = cirq.density_matrix_from_state_vector(other_args.target_tensor)
            other_args = cirq.ActOnDensityMatrixArgs(
                dm,
                [np.empty_like(dm) for _ in range(3)],
                dm.shape,
                other.prng,
                other.log_of_measurement_results,
                other.qubits,
                other.axes,
            )
        target.args = self_args.kronecker_product(other_args, inplace=self is target)

    def _on_factor(self, qubits, extracted, remainder, validate=False, atol=1e-07):
        extracted.args, remainder.args = self.args.factor(qubits, validate=validate)
        if len(qubits) == 1:
            state = [x != 0 for x in extracted.args._perform_measurement(qubits)]
            extracted.args = PureActOnArgs(
                state, extracted.args.qubits, extracted.args.log_of_measurement_results
            )

    def _on_transpose_to_qubit_order(
        self, qubits: Sequence['cirq.Qid'], target: 'ProgressiveActOnArgs'
    ):
        target.args = self.args.transpose_to_qubit_order(qubits)

class ProgressiveStepResult(cirq.StepResultBase[ProgressiveActOnArgs, ProgressiveActOnArgs]):
    def _simulator_state(self) -> ProgressiveActOnArgs:
        return self._merged_sim_state


class ProgressiveTrialResult(cirq.SimulationTrialResult):
    pass


class ProgressiveSimulator(
    cirq.SimulatorBase[
        ProgressiveStepResult, ProgressiveTrialResult, ProgressiveActOnArgs, ProgressiveActOnArgs
    ]
):
    def __init__(self, noise=None, split_untangled_states=True):
        super().__init__(
            noise=noise,
            split_untangled_states=split_untangled_states,
        )

    def _create_partial_act_on_args(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        logs: Dict[str, Any],
    ) -> ProgressiveActOnArgs:
        if initial_state is None:
            initial_state = 0
        bs = int_to_bool_list(initial_state, len(qubits))
        args = PureActOnArgs(bs, qubits, logs)
        return ProgressiveActOnArgs(args=args, qubits=qubits, logs=logs)

    def _create_simulator_trial_result(
        self,
        params: cirq.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_step_result: ProgressiveStepResult,
    ) -> ProgressiveTrialResult:
        return ProgressiveTrialResult(params, measurements, final_step_result=final_step_result)

    def _create_step_result(
        self,
        sim_state: cirq.OperationTarget[ProgressiveActOnArgs],
    ) -> ProgressiveStepResult:
        return ProgressiveStepResult(sim_state)


q0, q1 = cirq.LineQubit.range(2)
entangled_state_repr = np.array([[math.sqrt(0.5), 0], [0, math.sqrt(0.5)]])


class TestOp(cirq.Operation):
    def with_qubits(self, *new_qubits):
        pass

    @property
    def qubits(self):
        return [q0]


def test_simulate_empty_circuit():
    sim = ProgressiveSimulator()
    r = sim.simulate(cirq.Circuit())
    assert r._final_simulator_state.args.b == []


def test_X():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.X(q0),
    )

    clifford_simulator = ProgressiveSimulator()
    state_vector_simulator = cirq.Simulator()
    r = clifford_simulator.simulate(circuit)

    sv1 = r._final_simulator_state.args.state_vector()
    sv2 = state_vector_simulator.simulate(circuit).final_state_vector
    assert np.allclose(
        sv1,
        sv2,
    )


def test_XX():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.X(q0),
    )

    clifford_simulator = ProgressiveSimulator()
    state_vector_simulator = cirq.Simulator()
    r = clifford_simulator.simulate(circuit)

    sv1 = r._final_simulator_state.args.state_vector()
    sv2 = state_vector_simulator.simulate(circuit).final_state_vector
    assert np.allclose(
        sv1,
        sv2,
    )


def test_clifford_circuit_SHSYSHS():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.X(q0),
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
        cirq.Y(q0),
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
    )

    clifford_simulator = ProgressiveSimulator()
    state_vector_simulator = cirq.Simulator()
    r = clifford_simulator.simulate(circuit)

    sv2 = state_vector_simulator.simulate(circuit).final_state_vector
    sv1 = r._final_simulator_state.args.state.state_vector()
    assert np.allclose(
        sv1,
        sv2,
    )


def test_state_vector():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.X(q0),
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
        cirq.Y(q0),
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
        cirq.HPowGate(exponent=0.4).on(q0),
        cirq.X(q0),
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
        cirq.Y(q0),
        cirq.S(q0),
        cirq.H(q0),
        cirq.S(q0),
    )

    clifford_simulator = ProgressiveSimulator()
    state_vector_simulator = cirq.Simulator()
    r = clifford_simulator.simulate(circuit)

    sv2 = state_vector_simulator.simulate(circuit).final_state_vector
    sv1 = r._final_simulator_state.args.target_tensor
    assert np.allclose(
        sv1,
        sv2,
    )

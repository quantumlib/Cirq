# Copyright 2020 The Cirq Developers
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

import numpy as np
import pytest

import cirq


def test_init_state():
    args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(1), initial_state=1)
    np.testing.assert_allclose(args.state.state_vector(), [0, 1])
    with pytest.raises(ValueError, match='Must specify qubits'):
        _ = cirq.StabilizerChFormSimulationState(initial_state=1)


def test_cannot_act():
    class NoDetails(cirq.testing.SingleQubitGate):
        pass

    args = cirq.StabilizerChFormSimulationState(qubits=[], prng=np.random.RandomState())

    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(NoDetails(), args, qubits=())


def test_gate_with_act_on():
    class CustomGate(cirq.testing.SingleQubitGate):
        def _act_on_(self, sim_state, qubits):
            if isinstance(sim_state, cirq.StabilizerChFormSimulationState):
                qubit = sim_state.qubit_map[qubits[0]]
                sim_state.state.gamma[qubit] += 1
                return True

    state = cirq.StabilizerStateChForm(num_qubits=3)
    args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(3), prng=np.random.RandomState(), initial_state=state
    )

    cirq.act_on(CustomGate(), args, [cirq.LineQubit(1)])

    np.testing.assert_allclose(state.gamma, [0, 1, 0])


def test_unitary_fallback_y():
    class UnitaryYGate(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _unitary_(self):
            return np.array([[0, -1j], [1j, 0]])

    args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(3), prng=np.random.RandomState()
    )
    cirq.act_on(UnitaryYGate(), args, [cirq.LineQubit(1)])
    expected_args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(3), prng=np.random.RandomState()
    )
    cirq.act_on(cirq.Y, expected_args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(args.state.state_vector(), expected_args.state.state_vector())


def test_unitary_fallback_h():
    class UnitaryHGate(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _unitary_(self):
            return np.array([[1, 1], [1, -1]]) / (2**0.5)

    args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(3), prng=np.random.RandomState()
    )
    cirq.act_on(UnitaryHGate(), args, [cirq.LineQubit(1)])
    expected_args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(3), prng=np.random.RandomState()
    )
    cirq.act_on(cirq.H, expected_args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(args.state.state_vector(), expected_args.state.state_vector())


def test_copy():
    args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(3), prng=np.random.RandomState()
    )
    args1 = args.copy()
    assert isinstance(args1, cirq.StabilizerChFormSimulationState)
    assert args is not args1
    assert args.state is not args1.state
    np.testing.assert_equal(args.state.state_vector(), args1.state.state_vector())
    assert args.qubits == args1.qubits
    assert args.prng is args1.prng
    assert args.log_of_measurement_results is not args1.log_of_measurement_results
    assert args.log_of_measurement_results == args1.log_of_measurement_results

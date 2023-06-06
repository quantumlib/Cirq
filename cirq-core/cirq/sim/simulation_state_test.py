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

from typing import Any, Sequence

import numpy as np
import pytest

import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla


class DummyQuantumState(cirq.QuantumStateRepresentation):
    def copy(self, deep_copy_buffers=True):
        pass

    def measure(self, axes, seed=None):
        return [5, 3]

    def reindex(self, axes):
        return self

    def kron(self, other):
        return self

    def factor(self, axes, validate=True, atol=1e-07):
        return (self, self)


class DummySimulationState(cirq.SimulationState):
    def __init__(self, qubits=cirq.LineQubit.range(2)):
        super().__init__(state=DummyQuantumState(), qubits=qubits)

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        return True


class AncillaZ(cirq.Gate):
    def __init__(self, exponent=1):
        self._exponent = exponent

    def num_qubits(self) -> int:
        return 1

    def _decompose_(self, qubits):
        ancilla = cirq.NamedQubit('Ancilla')
        yield cirq.CX(qubits[0], ancilla)
        yield cirq.Z(ancilla) ** self._exponent
        yield cirq.CX(qubits[0], ancilla)


class AncillaH(cirq.Gate):
    def __init__(self, exponent=1):
        self._exponent = exponent

    def num_qubits(self) -> int:
        return 1

    def _decompose_(self, qubits):
        ancilla = cirq.NamedQubit('Ancilla')
        yield cirq.H(ancilla) ** self._exponent
        yield cirq.CX(ancilla, qubits[0])
        yield cirq.H(ancilla) ** self._exponent


class AncillaY(cirq.Gate):
    def __init__(self, exponent=1):
        self._exponent = exponent

    def num_qubits(self) -> int:
        return 1

    def _decompose_(self, qubits):
        ancilla = cirq.NamedQubit('Ancilla')
        yield cirq.Y(ancilla) ** self._exponent
        yield cirq.CX(ancilla, qubits[0])
        yield cirq.Y(ancilla) ** self._exponent


class DelegatingAncillaZ(cirq.Gate):
    def __init__(self, exponent=1):
        self._exponent = exponent

    def num_qubits(self) -> int:
        return 1

    def _decompose_(self, qubits):
        a = cirq.NamedQubit('a')
        yield cirq.CX(qubits[0], a)
        yield AncillaZ(self._exponent).on(a)
        yield cirq.CX(qubits[0], a)


def test_measurements():
    args = DummySimulationState()
    args.measure([cirq.LineQubit(0)], "test", [False], {})
    assert args.log_of_measurement_results["test"] == [5]


def test_decompose():
    class Composite(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)

    args = DummySimulationState()
    assert (
        simulation_state.strat_act_on_from_apply_decompose(Composite(), args, [cirq.LineQubit(0)])
        is NotImplemented
    )


def test_mapping():
    args = DummySimulationState()
    assert list(iter(args)) == cirq.LineQubit.range(2)
    r1 = args[cirq.LineQubit(0)]
    assert args is r1
    with pytest.raises(IndexError):
        _ = args[cirq.LineQubit(2)]


def test_swap_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = DummySimulationState()
    with pytest.raises(ValueError, match='Cannot swap different dimensions'):
        args.swap(q0, q1)


def test_rename_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = DummySimulationState()
    with pytest.raises(ValueError, match='Cannot rename to different dimensions'):
        args.rename(q0, q1)


def test_transpose_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    args = DummySimulationState()
    assert args.transpose_to_qubit_order((q1, q0)).qubits == (q1, q0)
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q2))
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q1, q1))


def test_field_getters():
    args = DummySimulationState()
    assert args.prng is np.random
    assert args.qubit_map == {q: i for i, q in enumerate(cirq.LineQubit.range(2))}


@pytest.mark.parametrize('exp', [-3, -2, -1, 0, 1, 2, 3])
def test_ancilla_z(exp):
    q = cirq.LineQubit(0)
    test_circuit = cirq.Circuit(AncillaZ(exp).on(q))

    control_circuit = cirq.Circuit(cirq.ZPowGate(exponent=exp).on(q))

    assert_test_circuit_for_sv_dm_simulators(test_circuit, control_circuit)


@pytest.mark.parametrize('exp', [-3, -2, -1, 0, 1, 2, 3])
def test_ancilla_y(exp):
    q = cirq.LineQubit(0)
    test_circuit = cirq.Circuit(AncillaY(exp).on(q))

    control_circuit = cirq.Circuit(cirq.Y(q))
    control_circuit.append(cirq.Y(q))
    control_circuit.append(cirq.XPowGate(exponent=exp).on(q))

    assert_test_circuit_for_sv_dm_simulators(test_circuit, control_circuit)


@pytest.mark.parametrize('exp', [-3, -2, -1, 0, 1, 2, 3])
def test_borrowable_qubit(exp):
    q = cirq.LineQubit(0)
    test_circuit = cirq.Circuit()
    test_circuit.append(cirq.H(q))
    test_circuit.append(cirq.X(q))
    test_circuit.append(AncillaH(exp).on(q))

    control_circuit = cirq.Circuit(cirq.H(q))

    assert_test_circuit_for_sv_dm_simulators(test_circuit, control_circuit)


@pytest.mark.parametrize('exp', [-3, -2, -1, 0, 1, 2, 3])
def test_delegating_gate_qubit(exp):
    q = cirq.LineQubit(0)

    test_circuit = cirq.Circuit()
    test_circuit.append(cirq.H(q))
    test_circuit.append(DelegatingAncillaZ(exp).on(q))

    control_circuit = cirq.Circuit(cirq.H(q))
    control_circuit.append(cirq.ZPowGate(exponent=exp).on(q))

    assert_test_circuit_for_sv_dm_simulators(test_circuit, control_circuit)


@pytest.mark.parametrize('num_ancilla', [1, 2, 3])
def test_phase_using_dirty_ancilla(num_ancilla: int):
    q = cirq.LineQubit(0)
    anc = cirq.NamedQubit.range(num_ancilla, prefix='anc')

    u = cirq.MatrixGate(cirq.testing.random_unitary(2 ** (num_ancilla + 1)))
    test_circuit = cirq.Circuit(
        u.on(q, *anc), PhaseUsingDirtyAncilla(ancilla_bitsize=num_ancilla).on(q)
    )
    control_circuit = cirq.Circuit(u.on(q, *anc), cirq.Z(q))
    assert_test_circuit_for_sv_dm_simulators(test_circuit, control_circuit)


@pytest.mark.parametrize('num_ancilla', [1, 2, 3])
@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 10))
def test_phase_using_clean_ancilla(num_ancilla: int, theta: float):
    q = cirq.LineQubit(0)
    u = cirq.MatrixGate(cirq.testing.random_unitary(2))
    test_circuit = cirq.Circuit(
        u.on(q), PhaseUsingCleanAncilla(theta=theta, ancilla_bitsize=num_ancilla).on(q)
    )
    control_circuit = cirq.Circuit(u.on(q), cirq.ZPowGate(exponent=theta).on(q))

    assert_test_circuit_for_sv_dm_simulators(test_circuit, control_circuit)


def assert_test_circuit_for_sv_dm_simulators(test_circuit, control_circuit) -> None:
    for test_simulator in ['cirq.final_state_vector', 'cirq.final_density_matrix']:
        test_sim = eval(test_simulator)(test_circuit)
        control_sim = eval(test_simulator)(control_circuit)
        assert np.allclose(test_sim, control_sim)

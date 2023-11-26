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


class ExampleQuantumState(cirq.QuantumStateRepresentation):
    def copy(self, deep_copy_buffers=True):
        pass

    def measure(self, axes, seed=None):
        return [5, 3]

    def reindex(self, axes):
        return self


class ExampleSimulationState(cirq.SimulationState):
    def __init__(self, qubits=cirq.LineQubit.range(2)):
        super().__init__(state=ExampleQuantumState(), qubits=qubits)

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        return True

    def add_qubits(self, qubits):
        ret = super().add_qubits(qubits)
        return self if NotImplemented else ret


class DelegatingAncillaZ(cirq.Gate):
    def __init__(self, exponent=1, measure_ancilla: bool = False):
        self._exponent = exponent
        self._measure_ancilla = measure_ancilla

    def num_qubits(self) -> int:
        return 1

    def _decompose_(self, qubits):
        a = cirq.NamedQubit('a')
        yield cirq.CX(qubits[0], a)
        yield PhaseUsingCleanAncilla(self._exponent).on(a)
        yield cirq.CX(qubits[0], a)
        if self._measure_ancilla:
            yield cirq.measure(a)


class Composite(cirq.Gate):
    def num_qubits(self) -> int:
        return 1

    def _decompose_(self, qubits):
        yield cirq.X(*qubits)


def test_measurements():
    args = ExampleSimulationState()
    args.measure([cirq.LineQubit(0)], "test", [False], {})
    assert args.log_of_measurement_results["test"] == [5]


def test_decompose():
    args = ExampleSimulationState()
    assert simulation_state.strat_act_on_from_apply_decompose(
        Composite(), args, [cirq.LineQubit(0)]
    )


def test_decompose_for_gate_allocating_qubits_raises():
    class Composite(cirq.testing.SingleQubitGate):
        def _decompose_(self, qubits):
            anc = cirq.NamedQubit("anc")
            yield cirq.CNOT(*qubits, anc)

    args = ExampleSimulationState()

    with pytest.raises(TypeError, match="add_qubits but not remove_qubits"):
        simulation_state.strat_act_on_from_apply_decompose(Composite(), args, [cirq.LineQubit(0)])


def test_mapping():
    args = ExampleSimulationState()
    assert list(iter(args)) == cirq.LineQubit.range(2)
    r1 = args[cirq.LineQubit(0)]
    assert args is r1
    with pytest.raises(IndexError):
        _ = args[cirq.LineQubit(2)]


def test_swap_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = ExampleSimulationState()
    with pytest.raises(ValueError, match='Cannot swap different dimensions'):
        args.swap(q0, q1)


def test_rename_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = ExampleSimulationState()
    with pytest.raises(ValueError, match='Cannot rename to different dimensions'):
        args.rename(q0, q1)


def test_transpose_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    args = ExampleSimulationState()
    assert args.transpose_to_qubit_order((q1, q0)).qubits == (q1, q0)
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q2))
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q1, q1))


def test_field_getters():
    args = ExampleSimulationState()
    assert args.prng is np.random
    assert args.qubit_map == {q: i for i, q in enumerate(cirq.LineQubit.range(2))}


@pytest.mark.parametrize('exp', np.linspace(0, 2 * np.pi, 10))
def test_delegating_gate_unitary(exp):
    q = cirq.LineQubit(0)

    test_circuit = cirq.Circuit()
    test_circuit.append(cirq.H(q))
    test_circuit.append(DelegatingAncillaZ(exp).on(q))

    control_circuit = cirq.Circuit(cirq.H(q))
    control_circuit.append(cirq.ZPowGate(exponent=exp).on(q))

    assert_test_circuit_for_dm_simulator(test_circuit, control_circuit)
    assert_test_circuit_for_sv_simulator(test_circuit, control_circuit)


@pytest.mark.parametrize('exp', np.linspace(0, 2 * np.pi, 10))
def test_delegating_gate_channel(exp):
    q = cirq.LineQubit(0)

    test_circuit = cirq.Circuit()
    test_circuit.append(cirq.H(q))
    test_circuit.append(DelegatingAncillaZ(exp, True).on(q))

    control_circuit = cirq.Circuit(cirq.H(q))
    control_circuit.append(cirq.ZPowGate(exponent=exp).on(q))

    assert_test_circuit_for_sv_simulator(test_circuit, control_circuit)
    assert_test_circuit_for_dm_simulator(test_circuit, control_circuit)


@pytest.mark.parametrize('num_ancilla', [1, 2, 3])
def test_phase_using_dirty_ancilla(num_ancilla: int):
    q = cirq.LineQubit(0)
    anc = cirq.NamedQubit.range(num_ancilla, prefix='anc')

    u = cirq.MatrixGate(cirq.testing.random_unitary(2 ** (num_ancilla + 1)))
    test_circuit = cirq.Circuit(
        u.on(q, *anc), PhaseUsingDirtyAncilla(ancilla_bitsize=num_ancilla).on(q)
    )
    control_circuit = cirq.Circuit(u.on(q, *anc), cirq.Z(q))
    assert_test_circuit_for_dm_simulator(test_circuit, control_circuit)
    assert_test_circuit_for_sv_simulator(test_circuit, control_circuit)


@pytest.mark.parametrize('num_ancilla', [1, 2, 3])
@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 10))
def test_phase_using_clean_ancilla(num_ancilla: int, theta: float):
    q = cirq.LineQubit(0)
    u = cirq.MatrixGate(cirq.testing.random_unitary(2))
    test_circuit = cirq.Circuit(
        u.on(q), PhaseUsingCleanAncilla(theta=theta, ancilla_bitsize=num_ancilla).on(q)
    )
    control_circuit = cirq.Circuit(u.on(q), cirq.ZPowGate(exponent=theta).on(q))
    assert_test_circuit_for_dm_simulator(test_circuit, control_circuit)
    assert_test_circuit_for_sv_simulator(test_circuit, control_circuit)


def assert_test_circuit_for_dm_simulator(test_circuit, control_circuit) -> None:
    # Density Matrix Simulator: For unitary gates, this fallbacks to `cirq.apply_channel`
    # which recursively calls to `cirq.apply_unitary(decompose=True)`.
    for split_untangled_states in [True, False]:
        sim = cirq.DensityMatrixSimulator(split_untangled_states=split_untangled_states)
        control_sim = sim.simulate(control_circuit).final_density_matrix
        test_sim = sim.simulate(test_circuit).final_density_matrix
        assert np.allclose(test_sim, control_sim)


def assert_test_circuit_for_sv_simulator(test_circuit, control_circuit) -> None:
    # State Vector Simulator.
    for split_untangled_states in [True, False]:
        sim = cirq.Simulator(split_untangled_states=split_untangled_states)
        control_sim = sim.simulate(control_circuit).final_state_vector
        test_sim = sim.simulate(test_circuit).final_state_vector
        assert np.allclose(test_sim, control_sim)

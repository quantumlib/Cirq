# Copyright 2022 The Cirq Developers
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

from typing import List, Sequence, Tuple

import numpy as np
import sympy

import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator


class ComputationalBasisState(cirq.qis.QuantumStateRepresentation):
    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool = True) -> 'ComputationalBasisState':
        return ComputationalBasisState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        return [self.basis[i] for i in axes]


class ComputationalBasisSimState(cirq.SimulationState[ComputationalBasisState]):
    def __init__(self, initial_state, qubits, classical_data):
        state = ComputationalBasisState(
            cirq.big_endian_int_to_bits(initial_state, bit_count=len(qubits))
        )
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence[cirq.Qid], allow_decompose: bool = True):
        gate = action.gate if isinstance(action, cirq.Operation) else action
        if isinstance(gate, cirq.XPowGate):
            i = self.qubit_map[qubits[0]]
            self._state.basis[i] = int(gate.exponent + self._state.basis[i]) % qubits[0].dimension
            return True


def create_test_circuit():
    q0, q1 = cirq.LineQid.range(2, dimension=3)
    x = cirq.XPowGate(dimension=3)
    return cirq.Circuit(
        x(q0),
        cirq.measure(q0, key='a'),
        x(q0).with_classical_controls('a'),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(x(q1), cirq.measure(q1, key='b')),
            repeat_until=cirq.SympyCondition(sympy.Eq(sympy.Symbol('b'), 2)),
            use_repetition_ids=False,
        ),
    )


def test_basis_state_simulator():
    sim = CustomStateSimulator(ComputationalBasisSimState)
    circuit = create_test_circuit()
    r = sim.simulate(circuit)
    assert r.measurements == {'a': np.array([1]), 'b': np.array([2])}
    assert r._final_simulator_state._state.basis == [2, 2]


def test_built_in_states():
    # Verify this works for the built-in states too, you just lose the custom step/trial results.
    sim = CustomStateSimulator(cirq.StateVectorSimulationState)
    circuit = create_test_circuit()
    r = sim.simulate(circuit)
    assert r.measurements == {'a': np.array([1]), 'b': np.array([2])}
    assert np.allclose(
        r._final_simulator_state._state._state_vector, [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
    )


def test_product_state_mode_built_in_state():
    sim = CustomStateSimulator(cirq.StateVectorSimulationState, split_untangled_states=True)
    circuit = create_test_circuit()
    r = sim.simulate(circuit)
    assert r.measurements == {'a': np.array([1]), 'b': np.array([2])}

    # Ensure the state is in product-state mode, and it's got three states (q0, q1, phase)
    assert isinstance(r._final_simulator_state, cirq.SimulationProductState)
    assert len(r._final_simulator_state.sim_states) == 3

    assert np.allclose(
        r._final_simulator_state.create_merged_state()._state._state_vector,
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
    )


def test_noise():
    x = cirq.XPowGate(dimension=3)
    sim = CustomStateSimulator(ComputationalBasisSimState, noise=x**2)
    circuit = create_test_circuit()
    r = sim.simulate(circuit)
    assert r.measurements == {'a': np.array([2]), 'b': np.array([2])}
    assert r._final_simulator_state._state.basis == [1, 2]


def test_run():
    sim = CustomStateSimulator(ComputationalBasisSimState)
    circuit = create_test_circuit()
    r = sim.run(circuit)
    assert np.allclose(r.records['a'], np.array([[1]]))
    assert np.allclose(r.records['b'], np.array([[1], [2]]))


def test_parameterized_repetitions():
    q = cirq.LineQid(0, dimension=5)
    x = cirq.XPowGate(dimension=5)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(x(q), cirq.measure(q, key='a')),
            repetitions=sympy.Symbol('r'),
            use_repetition_ids=False,
        )
    )

    sim = CustomStateSimulator(ComputationalBasisSimState)
    r = sim.run_sweep(circuit, [{'r': i} for i in range(1, 5)])
    assert np.allclose(r[0].records['a'], np.array([[1]]))
    assert np.allclose(r[1].records['a'], np.array([[1], [2]]))
    assert np.allclose(r[2].records['a'], np.array([[1], [2], [3]]))
    assert np.allclose(r[3].records['a'], np.array([[1], [2], [3], [4]]))


class ComputationalBasisProductState(cirq.qis.QuantumStateRepresentation):
    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool = True) -> 'ComputationalBasisProductState':
        return ComputationalBasisProductState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        return [self.basis[i] for i in axes]

    def kron(self, other: 'ComputationalBasisProductState') -> 'ComputationalBasisProductState':
        return ComputationalBasisProductState(self.basis + other.basis)

    def factor(
        self, axes: Sequence[int], *, validate=True, atol=1e-07
    ) -> Tuple['ComputationalBasisProductState', 'ComputationalBasisProductState']:
        extracted = ComputationalBasisProductState([self.basis[i] for i in axes])
        remainder = ComputationalBasisProductState(
            [self.basis[i] for i in range(len(self.basis)) if i not in axes]
        )
        return extracted, remainder

    def reindex(self, axes: Sequence[int]) -> 'ComputationalBasisProductState':
        return ComputationalBasisProductState([self.basis[i] for i in axes])

    @property
    def supports_factor(self) -> bool:
        return True


class ComputationalBasisSimProductState(cirq.SimulationState[ComputationalBasisProductState]):
    def __init__(self, initial_state, qubits, classical_data):
        state = ComputationalBasisProductState(
            cirq.big_endian_int_to_bits(initial_state, bit_count=len(qubits))
        )
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence[cirq.Qid], allow_decompose: bool = True):
        gate = action.gate if isinstance(action, cirq.Operation) else action
        if isinstance(gate, cirq.XPowGate):
            i = self.qubit_map[qubits[0]]
            self._state.basis[i] = int(gate.exponent + self._state.basis[i]) % qubits[0].dimension
            return True


def test_product_state_mode():
    sim = CustomStateSimulator(ComputationalBasisSimProductState, split_untangled_states=True)
    circuit = create_test_circuit()
    r = sim.simulate(circuit)
    assert r.measurements == {'a': np.array([1]), 'b': np.array([2])}

    # Ensure the state is in product-state mode, and it's got three states (q0, q1, phase)
    assert isinstance(r._final_simulator_state, cirq.SimulationProductState)
    assert len(r._final_simulator_state.sim_states) == 3
    assert r._final_simulator_state.create_merged_state()._state.basis == [2, 2]

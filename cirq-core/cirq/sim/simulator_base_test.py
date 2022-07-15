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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest
import sympy

import cirq


class CountingState(cirq.qis.QuantumStateRepresentation):
    def __init__(self, data, gate_count=0, measurement_count=0, copy_count=0):
        self.data = data
        self.gate_count = gate_count
        self.measurement_count = measurement_count
        self.copy_count = copy_count

    def measure(
        self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
    ) -> List[int]:
        self.measurement_count += 1
        return [self.gate_count]

    def kron(self: 'CountingState', other: 'CountingState') -> 'CountingState':
        return CountingState(
            self.data,
            self.gate_count + other.gate_count,
            self.measurement_count + other.measurement_count,
            self.copy_count + other.copy_count,
        )

    def factor(
        self: 'CountingState', axes: Sequence[int], *, validate=True, atol=1e-07
    ) -> Tuple['CountingState', 'CountingState']:
        return CountingState(
            self.data, self.gate_count, self.measurement_count, self.copy_count
        ), CountingState(self.data)

    def reindex(self: 'CountingState', axes: Sequence[int]) -> 'CountingState':
        return CountingState(self.data, self.gate_count, self.measurement_count, self.copy_count)

    def copy(self, deep_copy_buffers: bool = True) -> 'CountingState':
        return CountingState(
            self.data, self.gate_count, self.measurement_count, self.copy_count + 1
        )


class CountingSimulationState(cirq.SimulationState[CountingState]):
    def __init__(self, state, qubits, classical_data):
        state_obj = CountingState(state)
        super().__init__(state=state_obj, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        self._state.gate_count += 1
        return True

    @property
    def data(self):
        return self._state.data

    @property
    def gate_count(self):
        return self._state.gate_count

    @property
    def measurement_count(self):
        return self._state.measurement_count

    @property
    def copy_count(self):
        return self._state.copy_count


class SplittableCountingSimulationState(CountingSimulationState):
    @property
    def allows_factoring(self):
        return True


class CountingStepResult(cirq.StepResultBase[CountingSimulationState]):
    def sample(
        self,
        qubits: List[cirq.Qid],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> np.ndarray:
        measurements: List[List[int]] = []
        for _ in range(repetitions):
            measurements.append(self._merged_sim_state._perform_measurement(qubits))
        return np.array(measurements, dtype=int)

    def _simulator_state(self) -> CountingSimulationState:
        return self._merged_sim_state


class CountingTrialResult(cirq.SimulationTrialResultBase[CountingSimulationState]):
    pass


class CountingSimulator(
    cirq.SimulatorBase[CountingStepResult, CountingTrialResult, CountingSimulationState]
):
    def __init__(self, noise=None, split_untangled_states=False):
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)

    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: cirq.ClassicalDataStore,
    ) -> CountingSimulationState:
        return CountingSimulationState(
            qubits=qubits, state=initial_state, classical_data=classical_data
        )

    def _create_simulator_trial_result(
        self,
        params: cirq.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[CountingSimulationState]',
    ) -> CountingTrialResult:
        return CountingTrialResult(
            params, measurements, final_simulator_state=final_simulator_state
        )

    def _create_step_result(
        self, sim_state: cirq.SimulationStateBase[CountingSimulationState]
    ) -> CountingStepResult:
        return CountingStepResult(sim_state)


class SplittableCountingSimulator(CountingSimulator):
    def __init__(self, noise=None, split_untangled_states=True):
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)

    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: cirq.ClassicalDataStore,
    ) -> CountingSimulationState:
        return SplittableCountingSimulationState(
            qubits=qubits, state=initial_state, classical_data=classical_data
        )


q0, q1 = cirq.LineQubit.range(2)
entangled_state_repr = np.array([[math.sqrt(0.5), 0], [0, math.sqrt(0.5)]])


class TestOp(cirq.Operation):
    def with_qubits(self, *new_qubits):
        pass

    @property
    def qubits(self):
        return [q0]


def test_simulate_empty_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit())
    assert r._final_simulator_state.gate_count == 0
    assert r._final_simulator_state.measurement_count == 0
    assert r._final_simulator_state.copy_count == 0


def test_simulate_one_gate_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit(cirq.X(q0)))
    assert r._final_simulator_state.gate_count == 1
    assert r._final_simulator_state.copy_count == 0


def test_simulate_one_measurement_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit(cirq.measure(q0)))
    assert r._final_simulator_state.gate_count == 0
    assert r._final_simulator_state.measurement_count == 1
    assert r._final_simulator_state.copy_count == 0


def test_empty_circuit_simulation_has_moment():
    sim = CountingSimulator()
    steps = list(sim.simulate_moment_steps(cirq.Circuit()))
    assert len(steps) == 1


def test_noise_applied():
    sim = CountingSimulator(noise=cirq.X)
    r = sim.simulate(cirq.Circuit(cirq.X(q0)))
    assert r._final_simulator_state.gate_count == 2
    assert r._final_simulator_state.copy_count == 0


def test_noise_applied_measurement_gate():
    sim = CountingSimulator(noise=cirq.X)
    r = sim.simulate(cirq.Circuit(cirq.measure(q0)))
    assert r._final_simulator_state.gate_count == 1
    assert r._final_simulator_state.measurement_count == 1
    assert r._final_simulator_state.copy_count == 0


def test_parameterized_copies_all_but_last():
    sim = CountingSimulator()
    n = 4
    rs = sim.simulate_sweep(cirq.Circuit(cirq.X(q0) ** 'a'), [{'a': i} for i in range(n)])
    for i in range(n):
        r = rs[i]
        assert r._final_simulator_state.gate_count == 1
        assert r._final_simulator_state.measurement_count == 0
        assert r._final_simulator_state.copy_count == 0 if i == n - 1 else 1


def test_cannot_act():
    class BadOp(TestOp):
        def _act_on_(self, sim_state):
            raise TypeError()

    sim = CountingSimulator()
    with pytest.raises(TypeError, match="CountingSimulator doesn't support .*BadOp"):
        sim.simulate(cirq.Circuit(BadOp()))


def test_run_one_gate_circuit():
    sim = CountingSimulator()
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[1], [1]])


def test_run_one_gate_circuit_noise():
    sim = CountingSimulator(noise=cirq.X)
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[2], [2]])


def test_run_non_unitary_circuit():
    sim = CountingSimulator()
    r = sim.run(cirq.Circuit(cirq.phase_damp(1).on(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[1], [1]])


def test_run_non_unitary_circuit_non_unitary_state():
    class DensityCountingSimulator(CountingSimulator):
        def _can_be_in_run_prefix(self, val):
            return not cirq.is_measurement(val)

    sim = DensityCountingSimulator()
    r = sim.run(cirq.Circuit(cirq.phase_damp(1).on(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[1], [1]])


def test_run_non_terminal_measurement():
    sim = CountingSimulator()
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0), cirq.X(q0)), repetitions=2)
    assert np.allclose(r.measurements['q(0)'], [[1], [1]])


def test_integer_initial_state_is_split():
    sim = SplittableCountingSimulator()
    state = sim._create_simulation_state(2, (q0, q1))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0].data == 1
    assert state[q1].data == 0
    assert state[None].data == 0


def test_integer_initial_state_is_not_split_if_disabled():
    sim = SplittableCountingSimulator(split_untangled_states=False)
    state = sim._create_simulation_state(2, (q0, q1))
    assert isinstance(state, SplittableCountingSimulationState)
    assert state[q0] is state[q1]
    assert state.data == 2


def test_integer_initial_state_is_not_split_if_impossible():
    sim = CountingSimulator()
    state = sim._create_simulation_state(2, (q0, q1))
    assert isinstance(state, CountingSimulationState)
    assert not isinstance(state, SplittableCountingSimulationState)
    assert state[q0] is state[q1]
    assert state.data == 2


def test_non_integer_initial_state_is_not_split():
    sim = SplittableCountingSimulator()
    state = sim._create_simulation_state(entangled_state_repr, (q0, q1))
    assert len(set(state.values())) == 2
    assert (state[q0].data == entangled_state_repr).all()
    assert state[q1] is state[q0]
    assert state[None].data == 0


def test_entanglement_causes_join():
    sim = SplittableCountingSimulator()
    state = sim._create_simulation_state(2, (q0, q1))
    assert len(set(state.values())) == 3
    state.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(state.values())) == 2
    assert state[q0] is state[q1]
    assert state[None] is not state[q0]


def test_measurement_causes_split():
    sim = SplittableCountingSimulator()
    state = sim._create_simulation_state(entangled_state_repr, (q0, q1))
    assert len(set(state.values())) == 2
    state.apply_operation(cirq.measure(q0))
    assert len(set(state.values())) == 3
    assert state[q0] is not state[q1]
    assert state[q0] is not state[None]


def test_measurement_does_not_split_if_disabled():
    sim = SplittableCountingSimulator(split_untangled_states=False)
    state = sim._create_simulation_state(2, (q0, q1))
    assert isinstance(state, SplittableCountingSimulationState)
    state.apply_operation(cirq.measure(q0))
    assert isinstance(state, SplittableCountingSimulationState)
    assert state[q0] is state[q1]


def test_measurement_does_not_split_if_impossible():
    sim = CountingSimulator()
    state = sim._create_simulation_state(2, (q0, q1))
    assert isinstance(state, CountingSimulationState)
    assert not isinstance(state, SplittableCountingSimulationState)
    state.apply_operation(cirq.measure(q0))
    assert isinstance(state, CountingSimulationState)
    assert not isinstance(state, SplittableCountingSimulationState)
    assert state[q0] is state[q1]


def test_reorder_succeeds():
    sim = SplittableCountingSimulator()
    state = sim._create_simulation_state(entangled_state_repr, (q0, q1))
    reordered = state[q0].transpose_to_qubit_order([q1, q0])
    assert reordered.qubits == (q1, q0)


@pytest.mark.parametrize('split', [True, False])
def test_sim_state_instance_unchanged_during_normal_sim(split: bool):
    sim = SplittableCountingSimulator(split_untangled_states=split)
    state = sim._create_simulation_state(0, (q0, q1))
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.reset(q1))
    for step in sim.simulate_moment_steps(circuit, initial_state=state):
        assert step._sim_state is state
        assert (step._merged_sim_state is not state) == split


def test_measurements_retained_in_step_results():
    sim = SplittableCountingSimulator()
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'), cirq.measure(q0, key='b'), cirq.measure(q0, key='c')
    )
    iterator = sim.simulate_moment_steps(circuit)
    assert next(iterator).measurements.keys() == {'a'}
    assert next(iterator).measurements.keys() == {'a', 'b'}
    assert next(iterator).measurements.keys() == {'a', 'b', 'c'}
    assert not any(iterator)


def test_sweep_unparameterized_prefix_not_repeated_iff_unitary():
    q = cirq.LineQubit(0)

    class TestOp(cirq.Operation):
        def __init__(self, *, has_unitary: bool):
            self.count = 0
            self.has_unitary = has_unitary

        def _act_on_(self, sim_state):
            self.count += 1
            return True

        def with_qubits(self, qubits):
            pass

        @property
        def qubits(self):
            return (q,)

        def _has_unitary_(self):
            return self.has_unitary

    simulator = CountingSimulator()
    params = [cirq.ParamResolver({'a': 0}), cirq.ParamResolver({'a': 1})]

    op1 = TestOp(has_unitary=True)
    op2 = TestOp(has_unitary=True)
    circuit = cirq.Circuit(op1, cirq.XPowGate(exponent=sympy.Symbol('a'))(q), op2)
    rs = simulator.simulate_sweep(program=circuit, params=params)
    assert rs[0]._final_simulator_state.copy_count == 1
    assert rs[1]._final_simulator_state.copy_count == 0
    assert op1.count == 1
    assert op2.count == 2

    op1 = TestOp(has_unitary=False)
    op2 = TestOp(has_unitary=False)
    circuit = cirq.Circuit(op1, cirq.XPowGate(exponent=sympy.Symbol('a'))(q), op2)
    rs = simulator.simulate_sweep(program=circuit, params=params)
    assert rs[0]._final_simulator_state.copy_count == 1
    assert rs[1]._final_simulator_state.copy_count == 0
    assert op1.count == 2
    assert op2.count == 2


def test_inhomogeneous_measurement_count_padding():
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    sim = cirq.Simulator()
    c = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.X(q) ** 0.2, cirq.measure(q, key=key)),
            use_repetition_ids=False,
            repeat_until=cirq.KeyCondition(key),
        )
    )
    results = sim.run(c, repetitions=10)
    for i in range(10):
        assert np.sum(results.records['m'][i, :, :]) == 1

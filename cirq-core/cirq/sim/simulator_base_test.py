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
    def __init__(self, state, gate_count=0, measurement_count=0):
        self.state = state
        self.gate_count = gate_count
        self.measurement_count = measurement_count

    def measure(
        self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
    ) -> List[int]:
        self.measurement_count += 1
        return [self.gate_count]

    def kron(self: 'CountingState', other: 'CountingState') -> 'CountingState':
        return CountingState(
            self.state,
            self.gate_count + other.gate_count,
            self.measurement_count + other.measurement_count,
        )

    def factor(
        self: 'CountingState', axes: Sequence[int], *, validate=True, atol=1e-07
    ) -> Tuple['CountingState', 'CountingState']:
        return CountingState(self.state, self.gate_count, self.measurement_count), CountingState(
            self.state
        )

    def reindex(self: 'CountingState', axes: Sequence[int]) -> 'CountingState':
        return self.copy()

    def copy(self, deep_copy_buffers: bool = True) -> 'CountingState':
        return CountingState(
            state=self.state, gate_count=self.gate_count, measurement_count=self.measurement_count
        )


class CountingActOnArgs(cirq.ActOnArgs):
    def __init__(self, state, qubits, classical_data):
        state_obj = CountingState(state)
        super().__init__(
            state=state_obj,
            qubits=qubits,
            classical_data=classical_data,
        )
        self._state: CountingState = state_obj

    def _act_on_fallback_(
        self,
        action: Any,
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        self._state.gate_count += 1
        return True

    @property
    def state(self):
        return self._state.state

    @property
    def gate_count(self):
        return self._state.gate_count

    @property
    def measurement_count(self):
        return self._state.measurement_count


class SplittableCountingActOnArgs(CountingActOnArgs):
    @property
    def allows_factoring(self):
        return True


class CountingStepResult(cirq.StepResultBase[CountingActOnArgs, CountingActOnArgs]):
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

    def _simulator_state(self) -> CountingActOnArgs:
        return self._merged_sim_state


class CountingTrialResult(cirq.SimulationTrialResultBase[CountingActOnArgs, CountingActOnArgs]):
    pass


class CountingSimulator(
    cirq.SimulatorBase[
        CountingStepResult, CountingTrialResult, CountingActOnArgs, CountingActOnArgs
    ]
):
    def __init__(self, noise=None, split_untangled_states=False):
        super().__init__(
            noise=noise,
            split_untangled_states=split_untangled_states,
        )

    def _create_partial_act_on_args(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: cirq.ClassicalDataStore,
    ) -> CountingActOnArgs:
        return CountingActOnArgs(qubits=qubits, state=initial_state, classical_data=classical_data)

    def _create_simulator_trial_result(
        self,
        params: cirq.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_step_result: CountingStepResult,
    ) -> CountingTrialResult:
        return CountingTrialResult(params, measurements, final_step_result=final_step_result)

    def _create_step_result(
        self,
        sim_state: cirq.OperationTarget[CountingActOnArgs],
    ) -> CountingStepResult:
        return CountingStepResult(sim_state)


class SplittableCountingSimulator(CountingSimulator):
    def __init__(self, noise=None, split_untangled_states=True):
        super().__init__(
            noise=noise,
            split_untangled_states=split_untangled_states,
        )

    def _create_partial_act_on_args(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: cirq.ClassicalDataStore,
    ) -> CountingActOnArgs:
        return SplittableCountingActOnArgs(
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


def test_simulate_one_gate_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit(cirq.X(q0)))
    assert r._final_simulator_state.gate_count == 1


def test_simulate_one_measurement_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit(cirq.measure(q0)))
    assert r._final_simulator_state.gate_count == 0
    assert r._final_simulator_state.measurement_count == 1


def test_empty_circuit_simulation_has_moment():
    sim = CountingSimulator()
    steps = list(sim.simulate_moment_steps(cirq.Circuit()))
    assert len(steps) == 1


def test_noise_applied():
    sim = CountingSimulator(noise=cirq.X)
    r = sim.simulate(cirq.Circuit(cirq.X(q0)))
    assert r._final_simulator_state.gate_count == 2


def test_noise_applied_measurement_gate():
    sim = CountingSimulator(noise=cirq.X)
    r = sim.simulate(cirq.Circuit(cirq.measure(q0)))
    assert r._final_simulator_state.gate_count == 1
    assert r._final_simulator_state.measurement_count == 1


def test_cannot_act():
    class BadOp(TestOp):
        def _act_on_(self, args):
            raise TypeError()

    sim = CountingSimulator()
    with pytest.raises(TypeError, match="CountingSimulator doesn't support .*BadOp"):
        sim.simulate(cirq.Circuit(BadOp()))


def test_run_one_gate_circuit():
    sim = CountingSimulator()
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['0'], [[1], [1]])


def test_run_one_gate_circuit_noise():
    sim = CountingSimulator(noise=cirq.X)
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['0'], [[2], [2]])


def test_run_non_unitary_circuit():
    sim = CountingSimulator()
    r = sim.run(cirq.Circuit(cirq.phase_damp(1).on(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['0'], [[1], [1]])


def test_run_no_reuse_buffer_warning():
    # coverage: ignore
    class MockCountingActOnArgs(CountingActOnArgs):
        def copy(self) -> 'MockCountingActOnArgs':  # type: ignore
            return super().copy()  # type: ignore

    # coverage: ignore
    class MockCountingStepResult(cirq.StepResultBase[MockCountingActOnArgs, MockCountingActOnArgs]):
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

        def _simulator_state(self) -> MockCountingActOnArgs:
            return self._merged_sim_state

    class MockCountingTrialResult(
        cirq.SimulationTrialResultBase[MockCountingActOnArgs, MockCountingActOnArgs]
    ):
        pass

    # coverage: ignore
    class MockCountingSimulator(
        cirq.SimulatorBase[
            MockCountingStepResult,
            MockCountingTrialResult,
            MockCountingActOnArgs,
            MockCountingActOnArgs,
        ]
    ):
        def _create_partial_act_on_args(
            self,
            initial_state: Any,
            qubits: Sequence['cirq.Qid'],
            classical_data: cirq.ClassicalDataStore,
        ) -> MockCountingActOnArgs:
            return MockCountingActOnArgs(
                qubits=qubits, state=initial_state, classical_data=classical_data
            )

        def _create_simulator_trial_result(
            self,
            params: cirq.ParamResolver,
            measurements: Dict[str, np.ndarray],
            final_step_result: MockCountingStepResult,
        ) -> MockCountingTrialResult:
            return MockCountingTrialResult(
                params, measurements, final_step_result=final_step_result
            )

        def _create_step_result(
            self,
            sim_state: cirq.OperationTarget[MockCountingActOnArgs],
        ) -> MockCountingStepResult:
            return MockCountingStepResult(sim_state)

    sim = MockCountingSimulator()
    with cirq.testing.assert_deprecated('deep_copy_buffers', deadline='0.15'):
        sim.run(cirq.Circuit(cirq.phase_damp(1).on(q0), cirq.measure(q0)))


def test_run_non_unitary_circuit_non_unitary_state():
    class DensityCountingSimulator(CountingSimulator):
        def _can_be_in_run_prefix(self, val):
            return not cirq.is_measurement(val)

    sim = DensityCountingSimulator()
    r = sim.run(cirq.Circuit(cirq.phase_damp(1).on(q0), cirq.measure(q0)), repetitions=2)
    assert np.allclose(r.measurements['0'], [[1], [1]])


def test_run_non_terminal_measurement():
    sim = CountingSimulator()
    r = sim.run(cirq.Circuit(cirq.X(q0), cirq.measure(q0), cirq.X(q0)), repetitions=2)
    assert np.allclose(r.measurements['0'], [[1], [1]])


def test_integer_initial_state_is_split():
    sim = SplittableCountingSimulator()
    args = sim._create_act_on_args(2, (q0, q1))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0].state == 1
    assert args[q1].state == 0
    assert args[None].state == 0


def test_integer_initial_state_is_not_split_if_disabled():
    sim = SplittableCountingSimulator(split_untangled_states=False)
    args = sim._create_act_on_args(2, (q0, q1))
    assert isinstance(args, SplittableCountingActOnArgs)
    assert args[q0] is args[q1]
    assert args.state == 2


def test_integer_initial_state_is_not_split_if_impossible():
    sim = CountingSimulator()
    args = sim._create_act_on_args(2, (q0, q1))
    assert isinstance(args, CountingActOnArgs)
    assert not isinstance(args, SplittableCountingActOnArgs)
    assert args[q0] is args[q1]
    assert args.state == 2


def test_non_integer_initial_state_is_not_split():
    sim = SplittableCountingSimulator()
    args = sim._create_act_on_args(entangled_state_repr, (q0, q1))
    assert len(set(args.values())) == 2
    assert (args[q0].state == entangled_state_repr).all()
    assert args[q1] is args[q0]
    assert args[None].state == 0


def test_entanglement_causes_join():
    sim = SplittableCountingSimulator()
    args = sim._create_act_on_args(2, (q0, q1))
    assert len(set(args.values())) == 3
    args.apply_operation(cirq.CNOT(q0, q1))
    assert len(set(args.values())) == 2
    assert args[q0] is args[q1]
    assert args[None] is not args[q0]


def test_measurement_causes_split():
    sim = SplittableCountingSimulator()
    args = sim._create_act_on_args(entangled_state_repr, (q0, q1))
    assert len(set(args.values())) == 2
    args.apply_operation(cirq.measure(q0))
    assert len(set(args.values())) == 3
    assert args[q0] is not args[q1]
    assert args[q0] is not args[None]


def test_measurement_does_not_split_if_disabled():
    sim = SplittableCountingSimulator(split_untangled_states=False)
    args = sim._create_act_on_args(2, (q0, q1))
    assert isinstance(args, SplittableCountingActOnArgs)
    args.apply_operation(cirq.measure(q0))
    assert isinstance(args, SplittableCountingActOnArgs)
    assert args[q0] is args[q1]


def test_measurement_does_not_split_if_impossible():
    sim = CountingSimulator()
    args = sim._create_act_on_args(2, (q0, q1))
    assert isinstance(args, CountingActOnArgs)
    assert not isinstance(args, SplittableCountingActOnArgs)
    args.apply_operation(cirq.measure(q0))
    assert isinstance(args, CountingActOnArgs)
    assert not isinstance(args, SplittableCountingActOnArgs)
    assert args[q0] is args[q1]


def test_reorder_succeeds():
    sim = SplittableCountingSimulator()
    args = sim._create_act_on_args(entangled_state_repr, (q0, q1))
    reordered = args[q0].transpose_to_qubit_order([q1, q0])
    assert reordered.qubits == (q1, q0)


@pytest.mark.parametrize('split', [True, False])
def test_sim_state_instance_unchanged_during_normal_sim(split: bool):
    sim = SplittableCountingSimulator(split_untangled_states=split)
    args = sim._create_act_on_args(0, (q0, q1))
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.reset(q1))
    for step in sim.simulate_moment_steps(circuit, initial_state=args):
        assert step._sim_state is args
        assert (step._merged_sim_state is not args) == split


@pytest.mark.parametrize('split', [True, False])
def test_sim_state_instance_gets_changes_from_step_result(split: bool):
    sim = SplittableCountingSimulator(split_untangled_states=split)
    args = sim._create_act_on_args(0, (q0, q1))
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.reset(q1))
    for step in sim.simulate_moment_steps(circuit, initial_state=args):
        assert step._sim_state is args
        args = sim._create_act_on_args(0, (q0, q1))
        step._sim_state = args
        assert (step._merged_sim_state is not args) == split


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

        def _act_on_(self, args):
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
    params = [
        cirq.ParamResolver({'a': 0}),
        cirq.ParamResolver({'a': 1}),
    ]

    op1 = TestOp(has_unitary=True)
    op2 = TestOp(has_unitary=True)
    circuit = cirq.Circuit(op1, cirq.XPowGate(exponent=sympy.Symbol('a'))(q), op2)
    simulator.simulate_sweep(program=circuit, params=params)
    assert op1.count == 1
    assert op2.count == 2

    op1 = TestOp(has_unitary=False)
    op2 = TestOp(has_unitary=False)
    circuit = cirq.Circuit(op1, cirq.XPowGate(exponent=sympy.Symbol('a'))(q), op2)
    simulator.simulate_sweep(program=circuit, params=params)
    assert op1.count == 2
    assert op2.count == 2

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
from typing import List, Dict, Any, Sequence, Tuple, Union

import numpy as np
import pytest

import cirq


class CountingActOnArgs(cirq.ActOnArgs):
    gate_count = 0
    measurement_count = 0

    def __init__(self, state, qubits, logs):
        super().__init__(
            qubits=qubits,
            log_of_measurement_results=logs,
        )
        self.state = state

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        self.measurement_count += 1
        return [self.gate_count]

    def copy(self) -> 'CountingActOnArgs':
        args = CountingActOnArgs(
            qubits=self.qubits,
            logs=self.log_of_measurement_results.copy(),
            state=self.state,
        )
        args.gate_count = self.gate_count
        args.measurement_count = self.measurement_count
        return args

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        self.gate_count += 1
        return True

    def sample(self, qubits, repetitions=1, seed=None):
        pass


class SplittableCountingActOnArgs(CountingActOnArgs):
    def kronecker_product(
        self, other: 'SplittableCountingActOnArgs'
    ) -> 'SplittableCountingActOnArgs':
        args = SplittableCountingActOnArgs(
            qubits=self.qubits + other.qubits,
            logs=self.log_of_measurement_results,
            state=None,
        )
        args.gate_count = self.gate_count + other.gate_count
        args.measurement_count = self.measurement_count + other.measurement_count
        return args

    def factor(
        self,
        qubits: Sequence['cirq.Qid'],
        *,
        validate=True,
        atol=1e-07,
    ) -> Tuple['SplittableCountingActOnArgs', 'SplittableCountingActOnArgs']:
        extracted_args = SplittableCountingActOnArgs(
            qubits=qubits,
            logs=self.log_of_measurement_results,
            state=None,
        )
        extracted_args.gate_count = self.gate_count
        extracted_args.measurement_count = self.measurement_count
        remainder_args = SplittableCountingActOnArgs(
            qubits=tuple(q for q in self.qubits if q not in qubits),
            logs=self.log_of_measurement_results,
            state=None,
        )
        return extracted_args, remainder_args

    def transpose_to_qubit_order(
        self, qubits: Sequence['cirq.Qid']
    ) -> 'SplittableCountingActOnArgs':
        args = SplittableCountingActOnArgs(
            qubits=qubits,
            logs=self.log_of_measurement_results,
            state=self.state,
        )
        args.gate_count = self.gate_count
        args.measurement_count = self.measurement_count
        return args


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


class CountingTrialResult(cirq.SimulationTrialResult):
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
        logs: Dict[str, Any],
    ) -> CountingActOnArgs:
        return CountingActOnArgs(qubits=qubits, state=initial_state, logs=logs)

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
        logs: Dict[str, Any],
    ) -> CountingActOnArgs:
        return SplittableCountingActOnArgs(qubits=qubits, state=initial_state, logs=logs)


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

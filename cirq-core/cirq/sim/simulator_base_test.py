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
from typing import List, Dict, Any, Sequence, Tuple

import copy
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

    def _perform_measurement(self) -> List[int]:
        self.measurement_count += 1
        return [self.gate_count]

    def copy(self) -> 'CountingActOnArgs':
        args = copy.copy(self)
        args._log_of_measurement_results = self.log_of_measurement_results.copy()
        return args

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        self.gate_count += 1
        return True

    def sample(self, qubits, repetitions=1, seed=None) -> np.ndarray:
        pass


class SplittableCountingActOnArgs(CountingActOnArgs):
    def join(self, other: 'SplittableCountingActOnArgs') -> 'SplittableCountingActOnArgs':
        args = SplittableCountingActOnArgs(
            qubits=self.qubits + other.qubits,
            logs=self.log_of_measurement_results,
            state=None,
        )
        args.gate_count = self.gate_count + other.gate_count
        args.measurement_count = self.measurement_count + other.measurement_count
        return args

    def extract(
        self, qubits: Sequence['cirq.Qid']
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

    def reorder(self, qubits: Sequence['cirq.Qid']) -> 'SplittableCountingActOnArgs':
        args = SplittableCountingActOnArgs(
            qubits=qubits,
            logs=self.log_of_measurement_results,
            state=self.state,
        )
        args.gate_count = self.gate_count
        args.measurement_count = self.measurement_count
        return args

    def sample(
        self,
        qubits: Sequence[cirq.Qid],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> np.ndarray:
        measurements: List[List[int]] = []
        for _ in range(repetitions):
            measurements.append(self._perform_measurement())
        return np.array(measurements, dtype=int)


class CountingStepResult(cirq.MultiArgStepResult[CountingActOnArgs, CountingActOnArgs]):
    def __init__(
        self,
        sim_state: cirq.OperationTarget[CountingActOnArgs],
        qubits: Sequence[cirq.Qid],
    ):
        super().__init__(sim_state, qubits)

    def _simulator_state(self) -> CountingActOnArgs:
        return self.merged_sim_state


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

    def _create_act_on_arg(
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
        final_simulator_state: CountingActOnArgs,
    ) -> CountingTrialResult:
        return CountingTrialResult(params, measurements, final_simulator_state)

    def _create_step_result(
        self,
        sim_state: cirq.OperationTarget[CountingActOnArgs],
        qubits: Sequence['cirq.Qid'],
    ) -> CountingStepResult:
        return CountingStepResult(sim_state, qubits)


class SplittableCountingSimulator(CountingSimulator):
    def __init__(self, noise=None, split_untangled_states=True):
        super().__init__(
            noise=noise,
            split_untangled_states=split_untangled_states,
        )

    def _create_act_on_arg(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        logs: Dict[str, Any],
    ) -> CountingActOnArgs:
        return SplittableCountingActOnArgs(qubits=qubits, state=initial_state, logs=logs)


q0, q1 = cirq.LineQubit.range(2)


class TestOp(cirq.Operation):
    def with_qubits(self, *new_qubits):
        pass

    @property
    def qubits(self):
        return [q0]


def test_simulate_empty_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit())
    assert r._final_simulator_state is None


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
    assert args[q0].state == 1
    assert args[q1].state == 0
    assert args[None].state == 0


def test_integer_initial_state_is_not_split_if_disabled():
    sim = SplittableCountingSimulator(split_untangled_states=False)
    args = sim._create_act_on_args(2, (q0, q1))
    assert isinstance(args, SplittableCountingActOnArgs)
    assert args.state == 2


def test_integer_initial_state_is_not_split_if_impossible():
    sim = CountingSimulator()
    args = sim._create_act_on_args(2, (q0, q1))
    assert isinstance(args, CountingActOnArgs)
    assert not isinstance(args, SplittableCountingActOnArgs)
    assert args.state == 2


def test_non_integer_initial_state_is_not_split():
    sim = SplittableCountingSimulator()
    args = sim._create_act_on_args('state', (q0, q1))
    assert len(set(args.values())) == 2
    assert args[q0].state == 'state'
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
    args = sim._create_act_on_args('state', (q0, q1))
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


def test_measurement_does_not_split_if_impossible():
    sim = CountingSimulator()
    args = sim._create_act_on_args(2, (q0, q1))
    assert isinstance(args, CountingActOnArgs)
    assert not isinstance(args, SplittableCountingActOnArgs)
    args.apply_operation(cirq.measure(q0))
    assert isinstance(args, CountingActOnArgs)
    assert not isinstance(args, SplittableCountingActOnArgs)


def test_reorder_succeeds():
    sim = SplittableCountingSimulator()
    args = sim._create_act_on_args('state', (q0, q1))
    reordered = args[q0].reorder([q1, q0])
    assert reordered.qubits == (q1, q0)

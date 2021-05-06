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
from typing import List, Dict, Any, Sequence

import numpy as np
import pytest

import cirq


class CountingActOnArgs(cirq.ActOnArgs):
    gate_count = 0
    measurement_count = 0

    def _perform_measurement(self) -> List[int]:
        self.measurement_count += 1
        return [self.gate_count]

    def copy(self) -> 'CountingActOnArgs':
        args = CountingActOnArgs(
            qubits=self.qubits,
            axes=self.axes,
            prng=self.prng,
            log_of_measurement_results=self.log_of_measurement_results.copy(),
        )
        args.gate_count = self.gate_count
        args.measurement_count = self.measurement_count
        return args

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        self.gate_count += 1
        return True


class CountingStepResult(cirq.StepResult[CountingActOnArgs]):
    def __init__(
        self,
        sim_state: CountingActOnArgs,
        qubit_map: Dict[cirq.Qid, int],
    ):
        super().__init__(measurements=sim_state.log_of_measurement_results.copy())
        self.sim_state = sim_state
        self.qubit_map = qubit_map

    def sample(
        self,
        qubits: List[cirq.Qid],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> np.ndarray:
        measurements: List[List[int]] = []
        for _ in range(repetitions):
            measurements.append(self.sim_state._perform_measurement())
        return np.array(measurements, dtype=int)

    def _simulator_state(self) -> CountingActOnArgs:
        return self.sim_state


class CountingTrialResult(cirq.SimulationTrialResult):
    pass


class CountingSimulator(
    cirq.SimulatorBase[
        CountingStepResult, CountingTrialResult, CountingActOnArgs, CountingActOnArgs
    ]
):
    def _create_act_on_args(
        self,
        initial_state: Any,
        qubits: Sequence[cirq.Qid],
    ) -> CountingActOnArgs:
        return CountingActOnArgs(cirq.value.parse_random_state(0), qubits)

    def _create_simulator_trial_result(
        self,
        params: cirq.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: CountingActOnArgs,
    ) -> CountingTrialResult:
        return CountingTrialResult(params, measurements, final_simulator_state)

    def _create_step_result(
        self,
        sim_state: CountingActOnArgs,
        qubit_map: Dict[cirq.Qid, int],
    ) -> CountingStepResult:
        return CountingStepResult(sim_state, qubit_map)


class TestOp(cirq.Operation):
    def with_qubits(self, *new_qubits):
        pass

    @property
    def qubits(self):
        return [q0]


q0 = cirq.LineQubit(0)


def test_simulate_empty_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit())
    assert r._final_simulator_state.gate_count == 0


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

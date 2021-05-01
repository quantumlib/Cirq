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

import cirq
from cirq.sim.act_on_args import TSelf
from cirq.sim.simulator import TSimulatorState


class CountingStepResult(cirq.StepResult):
    def sample(
        self,
        qubits: List[cirq.Qid],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> np.ndarray:
        pass

    def _simulator_state(self) -> TSimulatorState:
        pass


class CountingTrialResult(cirq.SimulationTrialResult):
    pass


class CountingActOnArgs(cirq.ActOnArgs):
    def _perform_measurement(self) -> List[int]:
        pass

    def copy(self: TSelf) -> TSelf:
        pass


class CountingSimulator(
    cirq.SimulationEngine[CountingStepResult, CountingTrialResult, int, CountingActOnArgs]
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
        final_simulator_state: int,
    ) -> CountingTrialResult:
        return CountingTrialResult(params, measurements, final_simulator_state)

    def _create_step_result(
        self,
        sim_state: CountingActOnArgs,
        qubit_map: Dict[cirq.Qid, int],
    ) -> CountingStepResult:
        return CountingStepResult()


def test_empty_circuit():
    c = CountingSimulator()
    c.simulate()

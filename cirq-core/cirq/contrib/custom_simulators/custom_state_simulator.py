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

from typing import Any, Dict, Generic, Sequence, Type, TYPE_CHECKING

import numpy as np

from cirq import sim
from cirq.sim.simulation_state import TSimulationState

if TYPE_CHECKING:
    import cirq


class CustomStateStepResult(sim.StepResultBase[TSimulationState], Generic[TSimulationState]):
    """The step result provided by `CustomStateSimulator.simulate_moment_steps`."""


class CustomStateTrialResult(
    sim.SimulationTrialResultBase[TSimulationState], Generic[TSimulationState]
):
    """The trial result provided by `CustomStateSimulator.simulate`."""


class CustomStateSimulator(
    sim.SimulatorBase[
        CustomStateStepResult[TSimulationState],
        CustomStateTrialResult[TSimulationState],
        TSimulationState,
    ],
    Generic[TSimulationState],
):
    """A simulator that can be used to simulate custom states."""

    def __init__(
        self,
        state_type: Type[TSimulationState],
        *,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        split_untangled_states: bool = False,
    ):
        """Initializes a CustomStateSimulator.

        Args:
            state_type: The class that represents the simulation state this simulator should use.
            noise: The noise model used by the simulator.
            split_untangled_states: True to run the simulation as a product state. This is only
                supported if the `state_type` supports it via an implementation of `kron` and
                `factor` methods. Otherwise a runtime error will occur during simulation."""
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)
        self.state_type = state_type

    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[TSimulationState]',
    ) -> 'CustomStateTrialResult[TSimulationState]':
        return CustomStateTrialResult(
            params, measurements, final_simulator_state=final_simulator_state
        )

    def _create_step_result(
        self, sim_state: 'cirq.SimulationStateBase[TSimulationState]'
    ) -> 'CustomStateStepResult[TSimulationState]':
        return CustomStateStepResult(sim_state)

    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        classical_data: 'cirq.ClassicalDataStore',
    ) -> TSimulationState:
        return self.state_type(
            initial_state=initial_state, qubits=qubits, classical_data=classical_data
        )  # type: ignore[call-arg]

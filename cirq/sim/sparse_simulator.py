# Copyright 2018 The Cirq Developers
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
"""

"""


from typing import Dict, Iterator, List, Union

import numpy as np

from cirq import circuits, study, schedules, ops, protocols
from cirq.sim import simulator, state


class Simulator(simulator.SimulatesSamples,
                simulator.SimulatesFinalWaveFunction):

    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        """
        """
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        if circuit.are_all_measurements_terminal():
            return self._run_sweep_sample(resolved_circuit, repetitions)
        else:
            return self._run_sweep_repeat(resolved_circuit, repetitions)

    def _run_sweep_sample(
        self,
        circuit: circuits.Circuit,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        """
        """
        pass

    def _run_sweep_repeat(
        self,
        circuit: circuits.Circuit,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        """
        """
        all_step_results = self._base_iterator(
                circuit,
                qubit_order=ops.QubitOrder.DEFAULT,
                initial_state=0,
                perform_measurements=False)
        step_result = None
        for step_result in all_step_results:
            pass


    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable = study.ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0) -> List['SimulationTrialResult']:
        """
        """
        pass


    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray],
            perform_measurements: bool=True,
    ) -> Iterator['StepResult']:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
                circuit.all_qubits())
        qubit_map = {q: i for i, q in enumerate(reversed(qubits))}
        state = state.to_valid_state_vector(initial_state, len(qubits))

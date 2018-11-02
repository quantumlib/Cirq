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
from typing import Dict, List, Union

import numpy as np

from cirq import circuits, study, schedules, ops
from cirq.sim import simulator


class Simulator(simulator.SimulatesSamples,
                simulator.SimulatesFinalWaveFunction):

    def _run(self, circuit: circuits.Circuit,
            param_resolver: study.ParamResolver, repetitions: int) -> Dict[
        str, List[np.ndarray]]:
        pass

    def simulate_sweep(self,
            program: Union[circuits.Circuit, schedules.Schedule],
            params: study.Sweepable = study.ParamResolver({}),
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Union[int, np.ndarray] = 0) -> List[
        'SimulationTrialResult']:
        pass



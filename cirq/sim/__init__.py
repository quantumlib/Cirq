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

"""Base simulation classes and generic simulators."""

from cirq.sim.simulator import (
    SimulatesSamples,
    SimulationTrialResult,
    StepResult,
    SimulatesIntermediateWaveFunction,
    SimulatesFinalWaveFunction,
)

from cirq.sim.state import (
    dirac_notation,
    sample_state,
    to_valid_state_vector,
    validate_normalized_state,
)

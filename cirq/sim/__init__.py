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

from cirq.sim.density_matrix_utils import (
    measure_density_matrix,
    sample_density_matrix,
    to_valid_density_matrix,
    von_neumann_entropy,
)

from cirq.sim.density_matrix_simulator import (
    DensityMatrixSimulator,
    DensityMatrixSimulatorState,
    DensityMatrixStepResult,
    DensityMatrixTrialResult,
)

from cirq.sim.mux import (
    final_wavefunction,
    sample,
    sample_sweep,
)

from cirq.sim.simulator import (
    SimulatesFinalState,
    SimulatesIntermediateState,
    SimulatesSamples,
    SimulationTrialResult,
    StepResult,
)

from cirq.sim.sparse_simulator import (
    Simulator,
    SparseSimulatorStep,
)

from cirq.sim.wave_function_simulator import (
    SimulatesIntermediateWaveFunction,
    WaveFunctionSimulatorState,
    WaveFunctionStepResult,
    WaveFunctionTrialResult,
)

from cirq.sim.wave_function import (
    bloch_vector_from_state_vector,
    density_matrix_from_state_vector,
    dirac_notation,
    measure_state_vector,
    sample_state_vector,
    StateVectorMixin,
    to_valid_state_vector,
    validate_normalized_state,
)

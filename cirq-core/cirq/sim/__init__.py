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

"""Classes for circuit simulators and base implementations of these classes."""

from cirq.sim.clifford import (
    CliffordSimulator,
    CliffordSimulatorStepResult,
    CliffordState,
    CliffordTrialResult,
    CliffordTableauSimulationState,
    StabilizerChFormSimulationState,
    StabilizerSampler,
    StabilizerSimulationState,
    StabilizerStateChForm,
)

from cirq.sim.density_matrix_simulation_state import DensityMatrixSimulationState

from cirq.sim.density_matrix_simulator import (
    DensityMatrixSimulator,
    DensityMatrixStepResult,
    DensityMatrixTrialResult,
)

from cirq.sim.density_matrix_utils import measure_density_matrix, sample_density_matrix

from cirq.sim.mux import (
    CIRCUIT_LIKE,
    final_density_matrix,
    final_state_vector,
    sample,
    sample_sweep,
)

from cirq.sim.simulation_product_state import SimulationProductState

from cirq.sim.simulation_state import SimulationState

from cirq.sim.simulation_state_base import SimulationStateBase

from cirq.sim.simulator import (
    SimulatesAmplitudes,
    SimulatesExpectationValues,
    SimulatesFinalState,
    SimulatesIntermediateState,
    SimulatesSamples,
    SimulationTrialResult,
    StepResult,
)

from cirq.sim.simulator_base import SimulationTrialResultBase, SimulatorBase, StepResultBase

from cirq.sim.sparse_simulator import Simulator, SparseSimulatorStep

from cirq.sim.state_vector import measure_state_vector, sample_state_vector, StateVectorMixin

from cirq.sim.state_vector_simulation_state import StateVectorSimulationState

from cirq.sim.classical_simulator import ClassicalStateSimulator

from cirq.sim.state_vector_simulator import (
    SimulatesIntermediateStateVector,
    StateVectorStepResult,
    StateVectorTrialResult,
)

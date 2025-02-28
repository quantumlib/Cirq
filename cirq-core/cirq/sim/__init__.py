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
    CliffordSimulator as CliffordSimulator,
    CliffordSimulatorStepResult as CliffordSimulatorStepResult,
    CliffordState as CliffordState,
    CliffordTrialResult as CliffordTrialResult,
    CliffordTableauSimulationState as CliffordTableauSimulationState,
    StabilizerChFormSimulationState as StabilizerChFormSimulationState,
    StabilizerSampler as StabilizerSampler,
    StabilizerSimulationState as StabilizerSimulationState,
    StabilizerStateChForm as StabilizerStateChForm,
)

from cirq.sim.density_matrix_simulation_state import (
    DensityMatrixSimulationState as DensityMatrixSimulationState,
)

from cirq.sim.density_matrix_simulator import (
    DensityMatrixSimulator as DensityMatrixSimulator,
    DensityMatrixStepResult as DensityMatrixStepResult,
    DensityMatrixTrialResult as DensityMatrixTrialResult,
)

from cirq.sim.density_matrix_utils import (
    measure_density_matrix as measure_density_matrix,
    sample_density_matrix as sample_density_matrix,
)

from cirq.sim.mux import (
    CIRCUIT_LIKE as CIRCUIT_LIKE,
    final_density_matrix as final_density_matrix,
    final_state_vector as final_state_vector,
    sample as sample,
    sample_sweep as sample_sweep,
)

from cirq.sim.simulation_product_state import SimulationProductState as SimulationProductState

from cirq.sim.simulation_state import SimulationState as SimulationState

from cirq.sim.simulation_state_base import SimulationStateBase as SimulationStateBase

from cirq.sim.simulator import (
    SimulatesAmplitudes as SimulatesAmplitudes,
    SimulatesExpectationValues as SimulatesExpectationValues,
    SimulatesFinalState as SimulatesFinalState,
    SimulatesIntermediateState as SimulatesIntermediateState,
    SimulatesSamples as SimulatesSamples,
    SimulationTrialResult as SimulationTrialResult,
    StepResult as StepResult,
)

from cirq.sim.simulator_base import (
    SimulationTrialResultBase as SimulationTrialResultBase,
    SimulatorBase as SimulatorBase,
    StepResultBase as StepResultBase,
)

from cirq.sim.sparse_simulator import (
    Simulator as Simulator,
    SparseSimulatorStep as SparseSimulatorStep,
)

from cirq.sim.state_vector import (
    measure_state_vector as measure_state_vector,
    sample_state_vector as sample_state_vector,
    StateVectorMixin as StateVectorMixin,
)

from cirq.sim.state_vector_simulation_state import (
    StateVectorSimulationState as StateVectorSimulationState,
)

from cirq.sim.classical_simulator import ClassicalStateSimulator as ClassicalStateSimulator

from cirq.sim.state_vector_simulator import (
    SimulatesIntermediateStateVector as SimulatesIntermediateStateVector,
    StateVectorStepResult as StateVectorStepResult,
    StateVectorTrialResult as StateVectorTrialResult,
)

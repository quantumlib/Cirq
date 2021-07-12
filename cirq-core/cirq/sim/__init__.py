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
from typing import Tuple, Dict

from cirq.sim.act_on_args import (
    ActOnArgs,
)

from cirq.sim.act_on_args_container import (
    ActOnArgsContainer,
)

from cirq.sim.act_on_density_matrix_args import (
    ActOnDensityMatrixArgs,
)

from cirq.sim.act_on_state_vector_args import (
    ActOnStateVectorArgs,
)

from cirq.sim.density_matrix_utils import (
    measure_density_matrix,
    sample_density_matrix,
)

from cirq.sim.density_matrix_simulator import (
    DensityMatrixSimulator,
    DensityMatrixSimulatorState,
    DensityMatrixStepResult,
    DensityMatrixTrialResult,
)

from cirq.sim.operation_target import OperationTarget

from cirq.sim.mux import (
    CIRCUIT_LIKE,
    final_density_matrix,
    final_state_vector,
    sample,
    sample_sweep,
)

from cirq.sim.simulator import (
    SimulatesAmplitudes,
    SimulatesExpectationValues,
    SimulatesFinalState,
    SimulatesIntermediateState,
    SimulatesSamples,
    SimulationTrialResult,
    StepResult,
)

from cirq.sim.simulator_base import (
    StepResultBase,
    SimulatorBase,
)

from cirq.sim.sparse_simulator import (
    Simulator,
    SparseSimulatorStep,
)

from cirq.sim.state_vector_simulator import (
    SimulatesIntermediateStateVector,
    StateVectorSimulatorState,
    StateVectorStepResult,
    StateVectorTrialResult,
)

from cirq.sim.state_vector import (
    measure_state_vector,
    sample_state_vector,
    StateVectorMixin,
)

from cirq.sim.clifford import (
    ActOnCliffordTableauArgs,
    ActOnStabilizerCHFormArgs,
    StabilizerSampler,
    StabilizerStateChForm,
    CliffordSimulator,
    CliffordState,
    CliffordTableau,
    CliffordTrialResult,
    CliffordSimulatorStepResult,
)

# pylint: disable=wrong-import-order
import sys as _sys
from cirq._compat import deprecate_attributes as _deprecate_attributes

deprecated_constants: Dict[str, Tuple[str, str]] = {
    # currently none, you can use this to deprecate constants, for example like this:
    # 'STATE_VECTOR_LIKE': ('v0.9', 'Use cirq.STATE_VECTOR_LIKE instead'),
}
_sys.modules[__name__] = _deprecate_attributes(_sys.modules[__name__], deprecated_constants)

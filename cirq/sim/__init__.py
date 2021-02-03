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

from cirq.sim.act_on_state_vector_args import (
    ActOnStateVectorArgs,
)

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
    CIRCUIT_LIKE,
    final_density_matrix,
    final_state_vector,
    final_wavefunction,
    sample,
    sample_sweep,
)

from cirq.sim.simulator import (
    SimulatesAmplitudes,
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

from cirq.sim.state_vector_simulator import (
    SimulatesIntermediateStateVector,
    SimulatesIntermediateWaveFunction,
    StateVectorSimulatorState,
    StateVectorStepResult,
    StateVectorTrialResult,
    WaveFunctionSimulatorState,
    WaveFunctionStepResult,
    WaveFunctionTrialResult,
)

from cirq.sim.state_vector import (
    bloch_vector_from_state_vector,
    density_matrix_from_state_vector,
    dirac_notation,
    measure_state_vector,
    sample_state_vector,
    StateVectorMixin,
    to_valid_state_vector,
    validate_normalized_state,
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

# Deprecated
# pylint: disable=wrong-import-order

from cirq.qis import STATE_VECTOR_LIKE

import sys as _sys
from cirq._compat import wrap_module as _wrap_module

deprecated_constants = {
    'STATE_VECTOR_LIKE': ('v0.9', 'Use cirq.STATE_VECTOR_LIKE instead'),
}
_sys.modules[__name__] = _wrap_module(_sys.modules[__name__], deprecated_constants)

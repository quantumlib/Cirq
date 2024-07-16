# Copyright 2020 The Cirq Developers
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
"""Experiments and tools for characterizing quantum operations."""

from cirq.experiments.qubit_characterizations import (
    RandomizedBenchMarkResult,
    single_qubit_randomized_benchmarking,
    single_qubit_state_tomography,
    TomographyResult,
    two_qubit_randomized_benchmarking,
    two_qubit_state_tomography,
    parallel_single_qubit_randomized_benchmarking,
)

from cirq.experiments.fidelity_estimation import (
    hog_score_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity,
    log_xeb_fidelity,
    log_xeb_fidelity_from_probabilities,
    xeb_fidelity,
)

from cirq.experiments.purity_estimation import purity_from_probabilities

from cirq.experiments.random_quantum_circuit_generation import (
    GRID_ALIGNED_PATTERN,
    GRID_STAGGERED_PATTERN,
    HALF_GRID_STAGGERED_PATTERN,
    GridInteractionLayer,
    random_rotations_between_grid_interaction_layers_circuit,
)

from cirq.experiments.readout_confusion_matrix import (
    TensoredConfusionMatrices,
    measure_confusion_matrix,
)

from cirq.experiments.n_qubit_tomography import (
    get_state_tomography_data,
    state_tomography,
    StateTomographyExperiment,
)

from cirq.experiments.single_qubit_readout_calibration import (
    estimate_parallel_single_qubit_readout_errors,
    estimate_single_qubit_readout_errors,
    SingleQubitReadoutCalibrationResult,
)

from cirq.experiments.t1_decay_experiment import t1_decay, T1DecayResult

from cirq.experiments.t2_decay_experiment import t2_decay, T2DecayResult

from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions

from cirq.experiments.two_qubit_xeb import (
    InferredXEBResult,
    TwoQubitXEBResult,
    parallel_two_qubit_xeb,
    run_rb_and_xeb,
)

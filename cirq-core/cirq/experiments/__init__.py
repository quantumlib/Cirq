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
    RandomizedBenchMarkResult as RandomizedBenchMarkResult,
    single_qubit_randomized_benchmarking as single_qubit_randomized_benchmarking,
    single_qubit_state_tomography as single_qubit_state_tomography,
    TomographyResult as TomographyResult,
    two_qubit_randomized_benchmarking as two_qubit_randomized_benchmarking,
    two_qubit_state_tomography as two_qubit_state_tomography,
    parallel_single_qubit_randomized_benchmarking as parallel_single_qubit_randomized_benchmarking,
)

from cirq.experiments.fidelity_estimation import (
    hog_score_xeb_fidelity_from_probabilities as hog_score_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity_from_probabilities as linear_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity as linear_xeb_fidelity,
    log_xeb_fidelity as log_xeb_fidelity,
    log_xeb_fidelity_from_probabilities as log_xeb_fidelity_from_probabilities,
    xeb_fidelity as xeb_fidelity,
)

from cirq.experiments.purity_estimation import (
    purity_from_probabilities as purity_from_probabilities,
)

from cirq.experiments.random_quantum_circuit_generation import (
    # pylint: disable=line-too-long
    GRID_ALIGNED_PATTERN as GRID_ALIGNED_PATTERN,
    GRID_STAGGERED_PATTERN as GRID_STAGGERED_PATTERN,
    HALF_GRID_STAGGERED_PATTERN as HALF_GRID_STAGGERED_PATTERN,
    GridInteractionLayer as GridInteractionLayer,
    random_rotations_between_grid_interaction_layers_circuit as random_rotations_between_grid_interaction_layers_circuit,
)

from cirq.experiments.readout_confusion_matrix import (
    TensoredConfusionMatrices as TensoredConfusionMatrices,
    measure_confusion_matrix as measure_confusion_matrix,
)

from cirq.experiments.n_qubit_tomography import (
    get_state_tomography_data as get_state_tomography_data,
    state_tomography as state_tomography,
    StateTomographyExperiment as StateTomographyExperiment,
)

from cirq.experiments.single_qubit_readout_calibration import (
    estimate_parallel_single_qubit_readout_errors as estimate_parallel_single_qubit_readout_errors,
    estimate_single_qubit_readout_errors as estimate_single_qubit_readout_errors,
    SingleQubitReadoutCalibrationResult as SingleQubitReadoutCalibrationResult,
)

from cirq.experiments.t1_decay_experiment import (
    t1_decay as t1_decay,
    T1DecayResult as T1DecayResult,
)

from cirq.experiments.t2_decay_experiment import (
    t2_decay as t2_decay,
    T2DecayResult as T2DecayResult,
)

from cirq.experiments.xeb_fitting import (
    XEBPhasedFSimCharacterizationOptions as XEBPhasedFSimCharacterizationOptions,
)

from cirq.experiments.two_qubit_xeb import (
    InferredXEBResult as InferredXEBResult,
    TwoQubitXEBResult as TwoQubitXEBResult,
    parallel_two_qubit_xeb as parallel_two_qubit_xeb,
    run_rb_and_xeb as run_rb_and_xeb,
)

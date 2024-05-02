# Copyright 2024 The Cirq Developers
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
"""Randomized benchmarking (RB) and cross-entropy benchmarking (XEB) experiments."""


from cirq.experiments.randomized_and_cross_entropy_benchmarking.randomized_benchmarking import (
    RandomizedBenchMarkResult,
    single_qubit_randomized_benchmarking,
    two_qubit_randomized_benchmarking,
    parallel_single_qubit_randomized_benchmarking,
)


from cirq.experiments.randomized_and_cross_entropy_benchmarking.fidelity_estimation import (
    hog_score_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity_from_probabilities,
    linear_xeb_fidelity,
    log_xeb_fidelity,
    log_xeb_fidelity_from_probabilities,
    xeb_fidelity,
)

from cirq.experiments.randomized_and_cross_entropy_benchmarking.xeb_fitting import (
    XEBPhasedFSimCharacterizationOptions,
    benchmark_2q_xeb_fidelities,
    parameterize_circuit,
    SqrtISwapXEBOptions,
    before_and_after_characterization,
    exponential_decay,
    fit_exponential_decays,
    characterize_phased_fsim_parameters_with_xeb_by_pair,
)

from cirq.experiments.randomized_and_cross_entropy_benchmarking.two_qubit_xeb import (
    InferredXEBResult,
    TwoQubitXEBResult,
    parallel_two_qubit_xeb,
    run_rb_and_xeb,
)

from cirq.experiments.randomized_and_cross_entropy_benchmarking.grid_parallel_two_qubit_xeb import (
    GridParallelXEBMetadata,
)

from cirq.experiments.randomized_and_cross_entropy_benchmarking.xeb_sampling import (
    sample_2q_xeb_circuits,
)

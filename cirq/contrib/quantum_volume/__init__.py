# Copyright 2019 The Cirq Developers
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
"""Utilities running the Quantum Volume benchmark on devices"""

from cirq.contrib.quantum_volume.quantum_volume import (
    generate_model_circuit,
    compute_heavy_set,
    sample_heavy_set,
    compile_circuit,
    calculate_quantum_volume,
    QuantumVolumeResult,
    CompilationResult,
)

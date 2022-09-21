# Copyright 2021 The Cirq Developers
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

import numpy as np

import cirq

# yapf: disable
SWAP = np.array([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
# yapf: enable


def time_kak_decomposition(target):
    """Benchmark kak_decomposition
    kak_decomposition is benchmarked because it was historically slow.
    See https://github.com/quantumlib/Cirq/issues/3840 for status of other benchmarks.
    """
    cirq.kak_decomposition(target)


time_kak_decomposition.params = [cirq.IdentityGate(2), cirq.SWAP, cirq.ISWAP, cirq.CZ, cirq.CNOT]
time_kak_decomposition.param_names = ["gate"]  # type: ignore

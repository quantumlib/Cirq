# Copyright 2026 The Cirq Developers
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

import pytest

import cirq


@pytest.mark.benchmark(group="greedy_qubit_manager")
def test_greedy_qubit_manager(benchmark):
    qm = cirq.GreedyQubitManager(prefix="bench", size=100)

    def _f():
        qubits = qm.qalloc(100)
        qm.qfree(qubits)

    benchmark(_f)

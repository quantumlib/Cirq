# Copyright 2023 The Cirq Developers
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

import cirq
from cirq_ft.infra.qubit_manager import GreedyQubitManager


def test_deprecated():
    with cirq.testing.assert_deprecated("Use cirq.GreedyQubitManager instead", deadline="v1.4"):
        _ = GreedyQubitManager('ancilla')

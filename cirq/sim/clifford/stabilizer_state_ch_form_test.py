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

import cirq
import cirq.testing

# TODO: This and clifford tableau need tests.
# Github issue: https://github.com/quantumlib/Cirq/issues/3021


def test_deprecated():
    with cirq.testing.assert_logs('wave_function', 'state_vector',
                                  'deprecated'):
        _ = cirq.StabilizerStateChForm(initial_state=0,
                                       num_qubits=1).wave_function()

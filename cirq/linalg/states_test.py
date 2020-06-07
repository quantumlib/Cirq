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

import cirq
import cirq.testing


def test_deprecated():
    with cirq.testing.assert_logs('cirq.eye_tensor', 'deprecated'):
        _ = cirq.linalg.eye_tensor((1,), dtype=float)

    with cirq.testing.assert_logs('cirq.one_hot', 'deprecated'):
        _ = cirq.linalg.one_hot(shape=(1,), dtype=float)

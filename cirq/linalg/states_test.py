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
from cirq._compat_test import capture_logging


def test_deprecated():
    with capture_logging() as log:
        _ = cirq.linalg.eye_tensor((1,), dtype=float)
    assert len(log) == 1
    assert "cirq.eye_tensor" in log[0].getMessage()
    assert "deprecated" in log[0].getMessage()

    with capture_logging() as log:
        _ = cirq.linalg.one_hot(shape=(1,), dtype=float)
    assert len(log) == 1
    assert "cirq.one_hot" in log[0].getMessage()
    assert "deprecated" in log[0].getMessage()

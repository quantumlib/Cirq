# Copyright 2022 The Cirq Developers
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


def test_q() -> None:
    assert cirq.q(0) == cirq.LineQubit(0)
    assert cirq.q(1, 2) == cirq.GridQubit(1, 2)
    assert cirq.q("foo") == cirq.NamedQubit("foo")


def test_q_invalid() -> None:
    # Ignore static type errors so we can test runtime typechecks.
    with pytest.raises(ValueError):
        cirq.q([1, 2, 3])  # type: ignore[call-overload]
    with pytest.raises(ValueError):
        cirq.q(1, "foo")  # type: ignore[call-overload]

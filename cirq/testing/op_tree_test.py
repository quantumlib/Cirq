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
import pytest

import cirq
from cirq.testing import assert_equivalent_op_tree


def test_assert_equivalent_op_tree():
    assert_equivalent_op_tree([], [])
    a = cirq.NamedQubit("a")
    assert_equivalent_op_tree([cirq.X(a)], [cirq.X(a)])

    assert_equivalent_op_tree(cirq.Circuit([cirq.X(a)]), [cirq.X(a)])
    assert_equivalent_op_tree(cirq.Circuit([cirq.X(a)], cirq.Moment()), [cirq.X(a)])

    with pytest.raises(AssertionError):
        assert_equivalent_op_tree([cirq.X(a)], [])

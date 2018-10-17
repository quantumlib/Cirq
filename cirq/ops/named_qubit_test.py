# Copyright 2018 The Cirq Developers
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
from cirq.ops.named_qubit import _pad_digits


def test_named_qubit_str():
    q = cirq.NamedQubit('a')
    assert q.name == 'a'
    assert str(q) == 'a'


# Python 2 gives a different repr due to unicode strings being prefixed with u.
@cirq.testing.only_test_in_python3
def test_named_qubit_repr():
    q = cirq.NamedQubit('a')
    assert repr(q) == "cirq.NamedQubit('a')"


def test_named_qubit_order():
    order = cirq.testing.OrderTester()
    order.add_ascending(
        cirq.NamedQubit(''),
        cirq.NamedQubit('1'),
        cirq.NamedQubit('a'),
        cirq.NamedQubit('a00000000'),
        cirq.NamedQubit('a00000000:8'),
        cirq.NamedQubit('a9'),
        cirq.NamedQubit('a09'),
        cirq.NamedQubit('a10'),
        cirq.NamedQubit('a11'),
        cirq.NamedQubit('aa'),
        cirq.NamedQubit('ab'),
        cirq.NamedQubit('b'),
    )
    order.add_ascending_equivalence_group(
        cirq.NamedQubit('c'),
        cirq.NamedQubit('c'),
    )


def test_pad_digits():
    assert _pad_digits('') == ''
    assert _pad_digits('a') == 'a'
    assert _pad_digits('a0') == 'a00000000:1'
    assert _pad_digits('a00') == 'a00000000:2'
    assert _pad_digits('a1bc23') == 'a00000001:1bc00000023:2'
    assert _pad_digits('a9') == 'a00000009:1'
    assert _pad_digits('a09') == 'a00000009:2'
    assert _pad_digits('a00000000:8') == 'a00000000:8:00000008:1'

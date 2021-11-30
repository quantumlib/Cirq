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


def test_init():
    q = cirq.NamedQubit('a')
    assert q.name == 'a'

    q = cirq.NamedQid('a', dimension=3)
    assert q.name == 'a'
    assert q.dimension == 3


def test_named_qubit_str():
    q = cirq.NamedQubit('a')
    assert q.name == 'a'
    assert str(q) == 'a'
    qid = cirq.NamedQid('a', dimension=3)
    assert qid.name == 'a'
    assert str(qid) == 'a (d=3)'


def test_named_qubit_repr():
    q = cirq.NamedQubit('a')
    assert repr(q) == "cirq.NamedQubit('a')"
    qid = cirq.NamedQid('a', dimension=3)
    assert repr(qid) == "cirq.NamedQid('a', dimension=3)"


def test_named_qubit_order():
    order = cirq.testing.OrderTester()
    order.add_ascending(
        cirq.NamedQid('', dimension=1),
        cirq.NamedQubit(''),
        cirq.NamedQid('', dimension=3),
        cirq.NamedQid('1', dimension=1),
        cirq.NamedQubit('1'),
        cirq.NamedQid('1', dimension=3),
        cirq.NamedQid('a', dimension=1),
        cirq.NamedQubit('a'),
        cirq.NamedQid('a', dimension=3),
        cirq.NamedQid('a00000000', dimension=1),
        cirq.NamedQubit('a00000000'),
        cirq.NamedQid('a00000000', dimension=3),
        cirq.NamedQid('a00000000:8', dimension=1),
        cirq.NamedQubit('a00000000:8'),
        cirq.NamedQid('a00000000:8', dimension=3),
        cirq.NamedQid('a9', dimension=1),
        cirq.NamedQubit('a9'),
        cirq.NamedQid('a9', dimension=3),
        cirq.NamedQid('a09', dimension=1),
        cirq.NamedQubit('a09'),
        cirq.NamedQid('a09', dimension=3),
        cirq.NamedQid('a10', dimension=1),
        cirq.NamedQubit('a10'),
        cirq.NamedQid('a10', dimension=3),
        cirq.NamedQid('a11', dimension=1),
        cirq.NamedQubit('a11'),
        cirq.NamedQid('a11', dimension=3),
        cirq.NamedQid('aa', dimension=1),
        cirq.NamedQubit('aa'),
        cirq.NamedQid('aa', dimension=3),
        cirq.NamedQid('ab', dimension=1),
        cirq.NamedQubit('ab'),
        cirq.NamedQid('ab', dimension=3),
        cirq.NamedQid('b', dimension=1),
        cirq.NamedQubit('b'),
        cirq.NamedQid('b', dimension=3),
    )
    order.add_ascending_equivalence_group(
        cirq.NamedQubit('c'),
        cirq.NamedQubit('c'),
        cirq.NamedQid('c', dimension=2),
        cirq.NamedQid('c', dimension=2),
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


def test_named_qubit_range():
    qubits = cirq.NamedQubit.range(2, prefix='a')
    assert qubits == [cirq.NamedQubit('a0'), cirq.NamedQubit('a1')]

    qubits = cirq.NamedQubit.range(-1, 4, 2, prefix='a')
    assert qubits == [
        cirq.NamedQubit('a-1'),
        cirq.NamedQubit('a1'),
        cirq.NamedQubit('a3'),
    ]


def test_named_qid_range():
    qids = cirq.NamedQid.range(2, prefix='a', dimension=3)
    assert qids == [cirq.NamedQid('a0', dimension=3), cirq.NamedQid('a1', dimension=3)]

    qids = cirq.NamedQid.range(-1, 4, 2, prefix='a', dimension=3)
    assert qids == [
        cirq.NamedQid('a-1', dimension=3),
        cirq.NamedQid('a1', dimension=3),
        cirq.NamedQid('a3', dimension=3),
    ]

    qids = cirq.NamedQid.range(2, prefix='a', dimension=4)
    assert qids == [cirq.NamedQid('a0', dimension=4), cirq.NamedQid('a1', dimension=4)]

    qids = cirq.NamedQid.range(-1, 4, 2, prefix='a', dimension=4)
    assert qids == [
        cirq.NamedQid('a-1', dimension=4),
        cirq.NamedQid('a1', dimension=4),
        cirq.NamedQid('a3', dimension=4),
    ]


def test_to_json():
    assert cirq.NamedQubit('c')._json_dict_() == {
        'name': 'c',
    }

    assert cirq.NamedQid('c', dimension=3)._json_dict_() == {
        'name': 'c',
        'dimension': 3,
    }

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

import pytest

import cirq
import cirq.google.api.v2 as v2


def test_qubit_to_proto_id():
    assert v2.qubit_to_proto_id(cirq.GridQubit(1, 2)) == '1_2'
    assert v2.qubit_to_proto_id(cirq.GridQubit(10, 2)) == '10_2'
    assert v2.qubit_to_proto_id(cirq.GridQubit(-1, 2)) == '-1_2'
    assert v2.qubit_to_proto_id(cirq.LineQubit(1)) == '1'
    assert v2.qubit_to_proto_id(cirq.LineQubit(10)) == '10'
    assert v2.qubit_to_proto_id(cirq.LineQubit(-1)) == '-1'
    assert v2.qubit_to_proto_id(cirq.NamedQubit('named')) == 'named'


def test_to_proto_id_unsupport_qid():

    class ValidQubit(cirq.Qid):

        def __init__(self, name):
            self._name = name

        @property
        def dimension(self):
            pass

        def _comparison_key(self):
            pass

    with pytest.raises(ValueError, match='ValidQubit'):
        _ = v2.qubit_to_proto_id(ValidQubit('d'))


def test_grid_qubit_from_proto_id():
    assert v2.grid_qubit_from_proto_id('1_2') == cirq.GridQubit(1, 2)
    assert v2.grid_qubit_from_proto_id('10_2') == cirq.GridQubit(10, 2)
    assert v2.grid_qubit_from_proto_id('-1_2') == cirq.GridQubit(-1, 2)
    assert v2.grid_qubit_from_proto_id('q-1_2') == cirq.GridQubit(-1, 2)
    assert v2.grid_qubit_from_proto_id('q1_2') == cirq.GridQubit(1, 2)


def test_grid_qubit_from_proto_id_invalid():
    with pytest.raises(ValueError, match='3_3_3'):
        _ = v2.grid_qubit_from_proto_id('3_3_3')
    with pytest.raises(ValueError, match='a_2'):
        _ = v2.grid_qubit_from_proto_id('a_2')
    with pytest.raises(ValueError, match='q1_q2'):
        v2.grid_qubit_from_proto_id('q1_q2')
    with pytest.raises(ValueError, match='q-1_q2'):
        v2.grid_qubit_from_proto_id('q-1_q2')
    with pytest.raises(ValueError, match='-1_q2'):
        v2.grid_qubit_from_proto_id('-1_q2')


def test_line_qubit_from_proto_id():
    assert v2.line_qubit_from_proto_id('1') == cirq.LineQubit(1)
    assert v2.line_qubit_from_proto_id('10') == cirq.LineQubit(10)
    assert v2.line_qubit_from_proto_id('-1') == cirq.LineQubit(-1)


def test_line_qubit_from_proto_id_invalid():
    with pytest.raises(ValueError, match='abc'):
        _ = v2.line_qubit_from_proto_id('abc')


def test_named_qubit_from_proto_id():
    assert v2.named_qubit_from_proto_id('a') == cirq.NamedQubit('a')


def test_generic_qubit_from_proto_id():
    assert v2.qubit_from_proto_id('1_2') == cirq.GridQubit(1, 2)
    assert v2.qubit_from_proto_id('1') == cirq.LineQubit(1)
    assert v2.qubit_from_proto_id('a') == cirq.NamedQubit('a')

    # Despite the fact that int(1_2_3) = 123, only pure numbers are parsed into
    # LineQubits.
    assert v2.qubit_from_proto_id('1_2_3') == cirq.NamedQubit('1_2_3')

    # All non-int-parseable names are converted to NamedQubits.
    assert v2.qubit_from_proto_id('a') == cirq.NamedQubit('a')
    assert v2.qubit_from_proto_id('1_b') == cirq.NamedQubit('1_b')

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

import pytest

import cirq


def test_qasm_output_args_validate():
    args = cirq.QasmArgs(version='2.0')
    args.validate_version('2.0')

    with pytest.raises(ValueError):
        args.validate_version('2.1')


def test_qasm_output_args_format():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m_a = cirq.measure(a, key='meas_a')
    m_b = cirq.measure(b, key='meas_b')
    args = cirq.QasmArgs(
        precision=4,
        version='2.0',
        qubit_id_map={a: 'aaa[0]', b: 'bbb[0]'},
        meas_key_id_map={'meas_a': 'm_a', 'meas_b': 'm_b'},
    )

    assert args.format('_{0}_', a) == '_aaa[0]_'
    assert args.format('_{0}_', b) == '_bbb[0]_'

    assert args.format('_{0:meas}_', cirq.measurement_key_name(m_a)) == '_m_a_'
    assert args.format('_{0:meas}_', cirq.measurement_key_name(m_b)) == '_m_b_'

    assert args.format('_{0}_', 89.1234567) == '_89.1235_'
    assert args.format('_{0}_', 1.23) == '_1.23_'

    assert args.format('_{0:half_turns}_', 89.1234567) == '_pi*89.1235_'
    assert args.format('_{0:half_turns}_', 1.23) == '_pi*1.23_'

    assert args.format('_{0}_', 'other') == '_other_'


def test_multi_qubit_gate_validate():
    class Example(cirq.Gate):
        def _num_qubits_(self) -> int:
            return self._num_qubits

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

    a, b, c, d = cirq.LineQubit.range(4)

    g = Example(3)

    assert g.num_qubits() == 3
    g.validate_args([a, b, c])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([a])
    with pytest.raises(ValueError):
        g.validate_args([a, b])
    with pytest.raises(ValueError):
        g.validate_args([a, b, c, d])

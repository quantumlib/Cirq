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

import numpy as np
import pytest
import sympy

import cirq
import cirq.ionq as ionq


def test_serialize_empty_circuit_invalid():
    empty = cirq.Circuit()
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='empty'):
        _ = serializer.serialize(empty)


def test_serialize_not_line_qubits_invalid():
    q0 = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='NamedQubit'):
        _ = serializer.serialize(circuit)


def test_serialize_parameterized_invalid():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q)**(sympy.Symbol('x')))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='parameterized'):
        _ = serializer.serialize(circuit)


def test_serialize_implicit_num_qubits():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result['qubits'] == 3


def test_serialize_non_gate_op_invalid():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.ParallelGateOperation(cirq.X, [q0, q1]))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='ParallelGateOperation'):
        _ = serializer.serialize(circuit)


def test_serialize_negative_line_qubit_invalid():
    q0 = cirq.LineQubit(-1)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='-1'):
        _ = serializer.serialize(circuit)


def test_serialize_pow_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for name, gate in (('rx', cirq.X), ('ry', cirq.Y), ('rz', cirq.Z)):
        for exponent in (1.1, 0.6):
            circuit = cirq.Circuit((gate**exponent)(q0))
            result = serializer.serialize(circuit)
            assert result == {
                'qubits':
                1,
                'circuit': [{
                    'gate': name,
                    'targets': [0],
                    'rotation': exponent * np.pi
                }],
            }


def test_serialize_pauli_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.X, 'x'), (cirq.Y, 'y'), (cirq.Z, 'z')):
        circuit = cirq.Circuit(gate(q0))
        result = serializer.serialize(circuit)
        assert result == {
            'qubits': 1,
            'circuit': [{
                'gate': name,
                'targets': [0]
            }]
        }


def test_serialize_sqrt_x_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.X(q0)**(0.5))
    result = serializer.serialize(circuit)
    assert result == {'qubits': 1, 'circuit': [{'gate': 'v', 'targets': [0]}]}
    circuit = cirq.Circuit(cirq.X(q0)**(-0.5))
    result = serializer.serialize(circuit)
    assert result == {'qubits': 1, 'circuit': [{'gate': 'vi', 'targets': [0]}]}


def test_serialize_s_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0)**(0.5))
    result = serializer.serialize(circuit)
    assert result == {'qubits': 1, 'circuit': [{'gate': 's', 'targets': [0]}]}
    circuit = cirq.Circuit(cirq.Z(q0)**(-0.5))
    result = serializer.serialize(circuit)
    assert result == {'qubits': 1, 'circuit': [{'gate': 'si', 'targets': [0]}]}


def test_serialize_h_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.H(q0))
    result = serializer.serialize(circuit)
    assert result == {'qubits': 1, 'circuit': [{'gate': 'h', 'targets': [0]}]}

    with pytest.raises(ValueError, match=r'H\^0.5'):
        circuit = cirq.Circuit(cirq.H(q0)**0.5)
        _ = serializer.serialize(circuit)


def test_serialize_t_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0)**(0.25))
    result = serializer.serialize(circuit)
    assert result == {'qubits': 1, 'circuit': [{'gate': 't', 'targets': [0]}]}
    circuit = cirq.Circuit(cirq.Z(q0)**(-0.25))
    result = serializer.serialize(circuit)
    assert result == {'qubits': 1, 'circuit': [{'gate': 'ti', 'targets': [0]}]}


def test_serialize_parity_pow_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.XXPowGate, 'xx'), (cirq.YYPowGate, 'yy'),
                       (cirq.ZZPowGate, 'zz')):
        for exponent in (0.5, 1.0, 1.5):
            circuit = cirq.Circuit(gate(exponent=exponent)(q0, q1))
            result = serializer.serialize(circuit)
            assert result == {
                'qubits':
                2,
                'circuit': [{
                    'gate': name,
                    'targets': [0, 1],
                    'rotation': exponent * np.pi
                }]
            }


def test_serialize_cnot_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))
    result = serializer.serialize(circuit)
    assert result == {
        'qubits': 2,
        'circuit': [{
            'gate': 'cnot',
            'control': 0,
            'target': 1
        }]
    }

    with pytest.raises(ValueError, match=r'CNOT\*\*0.5'):
        circuit = cirq.Circuit(cirq.CNOT(q0, q1)**0.5)
        _ = serializer.serialize(circuit)


def test_serialize_swap_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.SWAP(q0, q1))
    result = serializer.serialize(circuit)
    assert result == {
        'qubits': 2,
        'circuit': [{
            'gate': 'swap',
            'targets': [0, 1]
        }]
    }

    with pytest.raises(ValueError, match=r'SWAP\*\*0.5'):
        circuit = cirq.Circuit(cirq.SWAP(q0, q1)**0.5)
        _ = serializer.serialize(circuit)


def test_serialize_not_serializable():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.PhasedISwapPowGate()(q0, q1))
    with pytest.raises(ValueError, match='PhasedISWAP'):
        _ = serializer.serialize(circuit)


def test_serialize_atol():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer(atol=1e-1)
    # Within tolerance given above this is an X gate.
    circuit = cirq.Circuit(cirq.X(q0)**1.09)
    result = serializer.serialize(circuit)
    assert result['circuit'][0]['gate'] == 'x'

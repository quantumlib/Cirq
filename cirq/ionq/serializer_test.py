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


def test_serialize_non_terminal_measurements():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='d'), cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='end of circuit'):
        _ = serializer.serialize(circuit)


def test_serialize_not_line_qubits_invalid():
    q0 = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='NamedQubit'):
        _ = serializer.serialize(circuit)


def test_serialize_parameterized_invalid():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q) ** (sympy.Symbol('x')))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='parameterized'):
        _ = serializer.serialize(circuit)


def test_serialize_implicit_num_qubits():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result.body['qubits'] == 3


def test_serialize_too_many_measurement_keys():
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.Circuit([cirq.measure(q) for q in qubits])
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='10'):
        _ = serializer.serialize(circuit)


def test_serialize_measurement_key_too_long():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='0123456789a'))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='0123456789a'):
        _ = serializer.serialize(circuit)


def test_serialize_measurement_too_many_qubits():
    qubits = cirq.LineQubit.range(18)
    # Here we see that a comma value separates list of numbers from 0 to 17 is longer than
    # the IonQ maximum of 18.
    assert len(','.join(str(x) for x in range(18))) == 43
    circuit = cirq.Circuit(cirq.measure(*qubits, key='x'))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='40'):
        _ = serializer.serialize(circuit)


def test_serialize_measurement_key_cannot_be_shots():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='shots'))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='shots'):
        _ = serializer.serialize(circuit)


def test_serialize_non_gate_op_invalid():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.GlobalPhaseOperation(1j))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='GlobalPhaseOperation'):
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
            circuit = cirq.Circuit((gate ** exponent)(q0))
            result = serializer.serialize(circuit)
            assert result == ionq.SerializedProgram(
                body={
                    'qubits': 1,
                    'circuit': [{'gate': name, 'targets': [0], 'rotation': exponent * np.pi}],
                },
                metadata={},
            )


def test_serialize_pauli_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.X, 'x'), (cirq.Y, 'y'), (cirq.Z, 'z')):
        circuit = cirq.Circuit(gate(q0))
        result = serializer.serialize(circuit)
        assert result == ionq.SerializedProgram(
            body={'qubits': 1, 'circuit': [{'gate': name, 'targets': [0]}]}, metadata={}
        )


def test_serialize_sqrt_x_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.X(q0) ** (0.5))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': [{'gate': 'v', 'targets': [0]}]}, metadata={}
    )
    circuit = cirq.Circuit(cirq.X(q0) ** (-0.5))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': [{'gate': 'vi', 'targets': [0]}]}, metadata={}
    )


def test_serialize_s_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0) ** (0.5))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': [{'gate': 's', 'targets': [0]}]}, metadata={}
    )
    circuit = cirq.Circuit(cirq.Z(q0) ** (-0.5))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': [{'gate': 'si', 'targets': [0]}]}, metadata={}
    )


def test_serialize_h_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.H(q0))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': [{'gate': 'h', 'targets': [0]}]}, metadata={}
    )

    with pytest.raises(ValueError, match=r'H\^0.5'):
        circuit = cirq.Circuit(cirq.H(q0) ** 0.5)
        _ = serializer.serialize(circuit)


def test_serialize_t_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0) ** (0.25))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': [{'gate': 't', 'targets': [0]}]}, metadata={}
    )
    circuit = cirq.Circuit(cirq.Z(q0) ** (-0.25))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': [{'gate': 'ti', 'targets': [0]}]}, metadata={}
    )


def test_serialize_parity_pow_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.XXPowGate, 'xx'), (cirq.YYPowGate, 'yy'), (cirq.ZZPowGate, 'zz')):
        for exponent in (0.5, 1.0, 1.5):
            circuit = cirq.Circuit(gate(exponent=exponent)(q0, q1))
            result = serializer.serialize(circuit)
            assert result == ionq.SerializedProgram(
                body={
                    'qubits': 2,
                    'circuit': [{'gate': name, 'targets': [0, 1], 'rotation': exponent * np.pi}],
                },
                metadata={},
            )


def test_serialize_cnot_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 2, 'circuit': [{'gate': 'cnot', 'control': 0, 'target': 1}]}, metadata={}
    )

    with pytest.raises(ValueError, match=r'CNOT\*\*0.5'):
        circuit = cirq.Circuit(cirq.CNOT(q0, q1) ** 0.5)
        _ = serializer.serialize(circuit)


def test_serialize_swap_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.SWAP(q0, q1))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 2, 'circuit': [{'gate': 'swap', 'targets': [0, 1]}]}, metadata={}
    )

    with pytest.raises(ValueError, match=r'SWAP\*\*0.5'):
        circuit = cirq.Circuit(cirq.SWAP(q0, q1) ** 0.5)
        _ = serializer.serialize(circuit)


def test_serialize_measurement_gate():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 1, 'circuit': []}, metadata={'tomyheart': '0'}
    )


def test_serialize_measurement_gate_target_order():
    q0, _, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q2, q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(
        body={'qubits': 3, 'circuit': []}, metadata={'tomyheart': '2,0'}
    )


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
    circuit = cirq.Circuit(cirq.X(q0) ** 1.09)
    result = serializer.serialize(circuit)
    assert result.body['circuit'][0]['gate'] == 'x'

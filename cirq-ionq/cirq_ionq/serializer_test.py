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

import json

import numpy as np
import pytest
import sympy

import cirq
import cirq_ionq as ionq
from cirq_ionq.ionq_exceptions import IonQSerializerMixedGatesetsException


def test_serialize_single_circuit_empty_circuit_invalid():
    empty = cirq.Circuit()
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='empty'):
        _ = serializer.serialize_single_circuit(empty)


def test_serialize_many_circuits_empty_circuit_invalid():
    empty = cirq.Circuit()
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='empty'):
        _ = serializer.serialize_many_circuits([empty])


def test_serialize_single_circuit_non_terminal_measurements():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='d'), cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='end of circuit'):
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_non_terminal_measurements():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='d'), cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='end of circuit'):
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_not_line_qubits_invalid():
    q0 = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='NamedQubit'):
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_not_line_qubits_invalid():
    q0 = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='NamedQubit'):
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_parameterized_invalid():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q) ** (sympy.Symbol('x')))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='parameterized'):
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_parameterized_invalid():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q) ** (sympy.Symbol('x')))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='parameterized'):
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_implicit_num_qubits():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize_single_circuit(circuit)
    assert result.body['qubits'] == 3


def test_serialize_many_circuits_implicit_num_qubits():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize_many_circuits([circuit])
    assert result.body['qubits'] == 3


def test_serialize_single_circuit_settings():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize_single_circuit(
        circuit, job_settings={"foo": "bar", "key": "heart"}
    )
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 3, 'circuit': [{'gate': 'x', 'targets': [2]}]},
        metadata={},
        settings={"foo": "bar", "key": "heart"},
    )


def test_serialize_many_circuits_settings():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize_many_circuits(
        [circuit], job_settings={"foo": "bar", "key": "heart"}
    )
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 3,
            'circuits': [{'circuit': [{'gate': 'x', 'targets': [2]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[3]'},
        settings={"foo": "bar", "key": "heart"},
    )


def test_serialize_single_circuit_non_gate_op_invalid():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CircuitOperation(cirq.FrozenCircuit()))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='CircuitOperation'):
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_non_gate_op_invalid():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CircuitOperation(cirq.FrozenCircuit()))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='CircuitOperation'):
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_negative_line_qubit_invalid():
    q0 = cirq.LineQubit(-1)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='-1'):
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_negative_line_qubit_invalid():
    q0 = cirq.LineQubit(-1)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='-1'):
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_pow_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for name, gate in (('rx', cirq.X), ('ry', cirq.Y), ('rz', cirq.Z)):
        for exponent in (1.1, 0.6):
            circuit = cirq.Circuit((gate**exponent)(q0))
            result = serializer.serialize_single_circuit(circuit)
            assert result == ionq.SerializedProgram(
                body={
                    'gateset': 'qis',
                    'qubits': 1,
                    'circuit': [{'gate': name, 'targets': [0], 'rotation': exponent * np.pi}],
                },
                metadata={},
                settings={},
            )


def test_serialize_many_circuits_pow_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for name, gate in (('rx', cirq.X), ('ry', cirq.Y), ('rz', cirq.Z)):
        for exponent in (1.1, 0.6):
            circuit = cirq.Circuit((gate**exponent)(q0))
            result = serializer.serialize_many_circuits([circuit])
            assert result == ionq.SerializedProgram(
                body={
                    'gateset': 'qis',
                    'qubits': 1,
                    'circuits': [
                        {'circuit': [{'gate': name, 'targets': [0], 'rotation': exponent * np.pi}]}
                    ],
                },
                metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
                settings={},
            )


def test_serialize_single_circuit_pauli_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.X, 'x'), (cirq.Y, 'y'), (cirq.Z, 'z')):
        circuit = cirq.Circuit(gate(q0))
        result = serializer.serialize_single_circuit(circuit)
        assert result == ionq.SerializedProgram(
            body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': name, 'targets': [0]}]},
            metadata={},
            settings={},
        )


def test_serialize_many_circuits_pauli_gates():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.X, 'x'), (cirq.Y, 'y'), (cirq.Z, 'z')):
        circuit = cirq.Circuit(gate(q0))
        result = serializer.serialize_many_circuits([circuit])
        assert result == ionq.SerializedProgram(
            body={
                'gateset': 'qis',
                'qubits': 1,
                'circuits': [{'circuit': [{'gate': name, 'targets': [0]}]}],
            },
            metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
            settings={},
        )


def test_serialize_single_circuit_sqrt_x_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.X(q0) ** (0.5))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 'v', 'targets': [0]}]},
        metadata={},
        settings={},
    )
    circuit = cirq.Circuit(cirq.X(q0) ** (-0.5))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 'vi', 'targets': [0]}]},
        metadata={},
        settings={},
    )


def test_serialize_many_circuits_sqrt_x_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.X(q0) ** (0.5))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 1,
            'circuits': [{'circuit': [{'gate': 'v', 'targets': [0]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
        settings={},
    )
    circuit = cirq.Circuit(cirq.X(q0) ** (-0.5))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 1,
            'circuits': [{'circuit': [{'gate': 'vi', 'targets': [0]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
        settings={},
    )


def test_serialize_single_circuit_s_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0) ** (0.5))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 's', 'targets': [0]}]},
        metadata={},
        settings={},
    )
    circuit = cirq.Circuit(cirq.Z(q0) ** (-0.5))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 'si', 'targets': [0]}]},
        metadata={},
        settings={},
    )


def test_serialize_many_circuits_s_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0) ** (0.5))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 1,
            'circuits': [{'circuit': [{'gate': 's', 'targets': [0]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
        settings={},
    )
    circuit = cirq.Circuit(cirq.Z(q0) ** (-0.5))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 1,
            'circuits': [{'circuit': [{'gate': 'si', 'targets': [0]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
        settings={},
    )


def test_serialize_single_circuit_h_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.H(q0))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 'h', 'targets': [0]}]},
        metadata={},
        settings={},
    )

    with pytest.raises(ValueError, match=r'H\*\*0.5'):
        circuit = cirq.Circuit(cirq.H(q0) ** 0.5)
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_h_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.H(q0))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 1,
            'circuits': [{'circuit': [{'gate': 'h', 'targets': [0]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
        settings={},
    )

    with pytest.raises(ValueError, match=r'H\*\*0.5'):
        circuit = cirq.Circuit(cirq.H(q0) ** 0.5)
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_t_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0) ** (0.25))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 't', 'targets': [0]}]},
        metadata={},
        settings={},
    )
    circuit = cirq.Circuit(cirq.Z(q0) ** (-0.25))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 'ti', 'targets': [0]}]},
        metadata={},
        settings={},
    )


def test_serialize_many_circuits_t_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0) ** (0.25))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 1,
            'circuits': [{'circuit': [{'gate': 't', 'targets': [0]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
        settings={},
    )
    circuit = cirq.Circuit(cirq.Z(q0) ** (-0.25))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 1,
            'circuits': [{'circuit': [{'gate': 'ti', 'targets': [0]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[1]'},
        settings={},
    )


def test_serialize_single_circuit_parity_pow_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.XXPowGate, 'xx'), (cirq.YYPowGate, 'yy'), (cirq.ZZPowGate, 'zz')):
        for exponent in (0.5, 1.0, 1.5):
            circuit = cirq.Circuit(gate(exponent=exponent)(q0, q1))
            result = serializer.serialize_single_circuit(circuit)
            assert result == ionq.SerializedProgram(
                body={
                    'gateset': 'qis',
                    'qubits': 2,
                    'circuit': [{'gate': name, 'targets': [0, 1], 'rotation': exponent * np.pi}],
                },
                metadata={},
                settings={},
            )


def test_serialize__many_circuits_parity_pow_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    for gate, name in ((cirq.XXPowGate, 'xx'), (cirq.YYPowGate, 'yy'), (cirq.ZZPowGate, 'zz')):
        for exponent in (0.5, 1.0, 1.5):
            circuit = cirq.Circuit(gate(exponent=exponent)(q0, q1))
            result = serializer.serialize_many_circuits([circuit])
            assert result == ionq.SerializedProgram(
                body={
                    'gateset': 'qis',
                    'qubits': 2,
                    'circuits': [
                        {
                            'circuit': [
                                {'gate': name, 'targets': [0, 1], 'rotation': exponent * np.pi}
                            ]
                        }
                    ],
                },
                metadata={'measurements': '[{}]', 'qubit_numbers': '[2]'},
                settings={},
            )


def test_serialize_single_circuit_cnot_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 2,
            'circuit': [{'gate': 'cnot', 'control': 0, 'target': 1}],
        },
        metadata={},
        settings={},
    )

    with pytest.raises(ValueError, match=r'CNOT\*\*0.5'):
        circuit = cirq.Circuit(cirq.CNOT(q0, q1) ** 0.5)
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_cnot_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 2,
            'circuits': [{'circuit': [{'gate': 'cnot', 'control': 0, 'target': 1}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[2]'},
        settings={},
    )

    with pytest.raises(ValueError, match=r'CNOT\*\*0.5'):
        circuit = cirq.Circuit(cirq.CNOT(q0, q1) ** 0.5)
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_swap_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.SWAP(q0, q1))
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'qis', 'qubits': 2, 'circuit': [{'gate': 'swap', 'targets': [0, 1]}]},
        metadata={},
        settings={},
    )

    with pytest.raises(ValueError, match=r'SWAP\*\*0.5'):
        circuit = cirq.Circuit(cirq.SWAP(q0, q1) ** 0.5)
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_swap_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.SWAP(q0, q1))
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'qis',
            'qubits': 2,
            'circuits': [{'circuit': [{'gate': 'swap', 'targets': [0, 1]}]}],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[2]'},
        settings={},
    )

    with pytest.raises(ValueError, match=r'SWAP\*\*0.5'):
        circuit = cirq.Circuit(cirq.SWAP(q0, q1) ** 0.5)
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_measurement_gate():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'native', 'qubits': 1, 'circuit': []},
        metadata={'measurement0': f'tomyheart{chr(31)}0'},
        settings={},
    )


def test_serialize_many_circuits_measurement_gate():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={'gateset': 'native', 'qubits': 1, 'circuits': [{'circuit': []}]},
        metadata={
            'measurements': '[{"measurement0": "tomyheart\\u001f0"}]',
            'qubit_numbers': '[1]',
        },
        settings={},
    )


def test_serialize_single_circuit_measurement_gate_target_order():
    q0, _, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q2, q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'native', 'qubits': 3, 'circuit': []},
        metadata={'measurement0': f'tomyheart{chr(31)}2,0'},
        settings={},
    )


def test_serialize_many_circuits_measurement_gate_target_order():
    q0, _, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q2, q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={'gateset': 'native', 'qubits': 3, 'circuits': [{'circuit': []}]},
        metadata={
            'measurements': '[{"measurement0": "tomyheart\\u001f2,0"}]',
            'qubit_numbers': '[3]',
        },
        settings={},
    )


def test_serialize_single_circuit_measurement_gate_split_across_dict():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='a' * 60))
    serializer = ionq.Serializer()
    result = serializer.serialize_single_circuit(circuit)
    assert result.metadata['measurement0'] == 'a' * 40
    assert result.metadata['measurement1'] == 'a' * 20 + f'{chr(31)}0'


def test_serialize_many_circuits_measurement_gate_split_across_dict():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='a' * 60))
    serializer = ionq.Serializer()
    result = serializer.serialize_many_circuits([circuit])
    measurements = json.loads(result.metadata['measurements'])
    assert measurements[0]['measurement0'] == 'a' * 40
    assert measurements[0]['measurement1'] == 'a' * 20 + f'{chr(31)}0'


def test_serialize_single_circuit_native_gates():
    q0, q1, q2 = cirq.LineQubit.range(3)
    gpi = ionq.GPIGate(phi=0.1).on(q0)
    gpi2 = ionq.GPI2Gate(phi=0.2).on(q1)
    ms = ionq.MSGate(phi0=0.4, phi1=0.5).on(q1, q2)
    zz = ionq.ZZGate(theta=0.6).on(q1, q2)
    circuit = cirq.Circuit([gpi, gpi2, ms, zz])
    serializer = ionq.Serializer()
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'native',
            'qubits': 3,
            'circuit': [
                {'gate': 'gpi', 'target': 0, 'phase': 0.1},
                {'gate': 'gpi2', 'target': 1, 'phase': 0.2},
                {'gate': 'ms', 'targets': [1, 2], 'phases': [0.4, 0.5], 'angle': 0.25},
                {'gate': 'zz', 'targets': [1, 2], 'phase': 0.6},
            ],
        },
        metadata={},
        settings={},
    )


def test_serialize_many_circuits_native_gates():
    q0, q1, q2 = cirq.LineQubit.range(3)
    gpi = ionq.GPIGate(phi=0.1).on(q0)
    gpi2 = ionq.GPI2Gate(phi=0.2).on(q1)
    ms = ionq.MSGate(phi0=0.3, phi1=0.4).on(q1, q2)
    circuit = cirq.Circuit([gpi, gpi2, ms])
    serializer = ionq.Serializer()
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={
            'gateset': 'native',
            'qubits': 3,
            'circuits': [
                {
                    'circuit': [
                        {'gate': 'gpi', 'target': 0, 'phase': 0.1},
                        {'gate': 'gpi2', 'target': 1, 'phase': 0.2},
                        {'gate': 'ms', 'targets': [1, 2], 'phases': [0.3, 0.4], 'angle': 0.25},
                    ]
                }
            ],
        },
        metadata={'measurements': '[{}]', 'qubit_numbers': '[3]'},
        settings={},
    )


def test_serialize_many_circuits_raises_exception_on_mixed_native_and_qis_gates():
    (q1,) = cirq.LineQubit.range(1)
    (q2,) = cirq.LineQubit.range(1)
    gpi = ionq.GPIGate(phi=0.1).on(q1)
    x = cirq.X(q2)
    circuit1 = cirq.Circuit([gpi])
    circuit2 = cirq.Circuit([x])
    serializer = ionq.Serializer()
    with pytest.raises(IonQSerializerMixedGatesetsException) as exc_info:
        serializer.serialize_many_circuits([circuit1, circuit2])
    exception_message = (
        "For batch circuit submission, all circuits in a batch must "
        "contain the same type of gates: either 'qis' or 'native' gates."
    )

    assert exception_message in str(exc_info.value)


def test_serialize_single_circuit_measurement_gate_multiple_keys():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, key='b'))
    serializer = ionq.Serializer()
    result = serializer.serialize_single_circuit(circuit)
    assert result == ionq.SerializedProgram(
        body={'gateset': 'native', 'qubits': 2, 'circuit': []},
        metadata={'measurement0': f'a{chr(31)}0{chr(30)}b{chr(31)}1'},
        settings={},
    )


def test_serialize_many_circuits_measurement_gate_multiple_keys():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, key='b'))
    serializer = ionq.Serializer()
    result = serializer.serialize_many_circuits([circuit])
    assert result == ionq.SerializedProgram(
        body={'gateset': 'native', 'qubits': 2, 'circuits': [{'circuit': []}]},
        metadata={
            'measurements': '[{"measurement0": "a\\u001f0\\u001eb\\u001f1"}]',
            'qubit_numbers': '[2]',
        },
        settings={},
    )


def test_serialize_single_circuit_measurement_string_too_long():
    q = cirq.LineQubit(0)
    # Max limit for metadata is 9 keys of length 40.  Here we create a key of length
    # 40 * 9 - 1. When combined with one qubit for the qubit, 0, and the deliminator this
    # is just too big to fix.
    circuit = cirq.Circuit(cirq.measure(q, key='x' * (40 * 9 - 1)))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='too long'):
        _ = serializer.serialize_single_circuit(circuit)
    # Check that one fewer character is fine.
    circuit = cirq.Circuit(cirq.measure(q, key='x' * (40 * 9 - 2)))
    serializer = ionq.Serializer()
    _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_measurement_string_too_long():
    q = cirq.LineQubit(0)
    # Max limit for metadata is 9 keys of length 40.  Here we create a key of length
    # 40 * 9 - 1. When combined with one qubit for the qubit, 0, and the deliminator this
    # is just too big to fix.
    circuit = cirq.Circuit(cirq.measure(q, key='x' * (40 * 9 - 1)))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='too long'):
        _ = serializer.serialize_many_circuits([circuit])
    # Check that one fewer character is fine.
    circuit = cirq.Circuit(cirq.measure(q, key='x' * (40 * 9 - 2)))
    serializer = ionq.Serializer()
    _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_measurement_key_cannot_be_deliminator():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.measure(q0, key=f'ab{chr(30)}'))
    with pytest.raises(ValueError, match=f'ab{chr(30)}'):
        _ = serializer.serialize_single_circuit(circuit)
    circuit = cirq.Circuit(cirq.measure(q0, key=f'ab{chr(31)}'))
    with pytest.raises(ValueError, match=f'ab{chr(31)}'):
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_measurement_key_cannot_be_deliminator():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.measure(q0, key=f'ab{chr(30)}'))
    with pytest.raises(ValueError, match=f'ab{chr(30)}'):
        _ = serializer.serialize_many_circuits([circuit])
    circuit = cirq.Circuit(cirq.measure(q0, key=f'ab{chr(31)}'))
    with pytest.raises(ValueError, match=f'ab{chr(31)}'):
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_not_serializable():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.PhasedISwapPowGate()(q0, q1))
    with pytest.raises(ValueError, match='PhasedISWAP'):
        _ = serializer.serialize_single_circuit(circuit)


def test_serialize_many_circuits_not_serializable():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.PhasedISwapPowGate()(q0, q1))
    with pytest.raises(ValueError, match='PhasedISWAP'):
        _ = serializer.serialize_many_circuits([circuit])


def test_serialize_single_circuit_atol():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer(atol=1e-1)
    # Within tolerance given above this is an X gate.
    circuit = cirq.Circuit(cirq.X(q0) ** 1.09)
    result = serializer.serialize_single_circuit(circuit)
    assert result.body['circuit'][0]['gate'] == 'x'


def test_serialize_many_circuits_atol():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer(atol=1e-1)
    # Within tolerance given above this is an X gate.
    circuit = cirq.Circuit(cirq.X(q0) ** 1.09)
    result = serializer.serialize_many_circuits([circuit])
    assert result.body['circuits'][0]['circuit'][0]['gate'] == 'x'

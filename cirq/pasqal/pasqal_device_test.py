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

from cirq.pasqal import PasqalDevice


def generic_device(num_qubits) -> PasqalDevice:
    return PasqalDevice(qubits=cirq.NamedQubit.range(num_qubits, prefix='q'))


def test_init():
    d = generic_device(3)
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')

    assert d.qubit_set() == {q0, q1, q2}
    assert d.qubit_list() == [q0, q1, q2]
    assert d.supported_qubit_type == (cirq.NamedQubit,)


def test_init_errors():
    line = cirq.devices.LineQubit.range(3)
    with pytest.raises(TypeError, match="Unsupported qubit type"):
        PasqalDevice(qubits=line)
    with pytest.raises(ValueError, match="Too many qubits"):
        generic_device(101)


def test_decompose_error():
    d = generic_device(3)
    op = (cirq.ops.CCZ**1.5).on(*(d.qubit_list()))
    assert d.decompose_operation(op) == [op]

    op = cirq.H.on(cirq.NamedQubit('q0'))
    decomposition = d.decompose_operation(op)
    assert len(decomposition) == 2
    assert decomposition == [
        (cirq.Y**0.5).on(cirq.NamedQubit('q0')),
        cirq.XPowGate(exponent=1.0,
                      global_shift=-0.25).on(cirq.NamedQubit('q0'))
    ]

    # MeasurementGate is not a GateOperation
    with pytest.raises(TypeError):
        d.decompose_operation(cirq.ops.MeasurementGate(num_qubits=2))
    # It has to be made into one
    assert d.is_pasqal_device_op(
        cirq.ops.GateOperation(cirq.ops.MeasurementGate(2),
                               [cirq.NamedQubit('q0'),
                                cirq.NamedQubit('q1')]))

    assert PasqalDevice.is_pasqal_device_op(cirq.ops.X(cirq.NamedQubit('q0')))


def test_validate_operation_errors():
    d = generic_device(3)
    circuit = cirq.Circuit(device=d)

    with pytest.raises(ValueError, match="Unsupported operation"):
        d.validate_operation(cirq.NamedQubit('q0'))

    with pytest.raises(ValueError, match="is not a supported gate"):
        d.validate_operation((cirq.ops.H**0.2).on(cirq.NamedQubit('q0')))

    with pytest.raises(ValueError,
                       match="is not a valid qubit for gate cirq.X"):
        d.validate_operation(cirq.X.on(cirq.LineQubit(0)))

    with pytest.raises(ValueError, match="is not part of the device."):
        d.validate_operation(cirq.X.on(cirq.NamedQubit('q6')))

    with pytest.raises(NotImplementedError,
                       match="Measurements on Pasqal devices "
                       "don't support invert_mask."):
        circuit.append(cirq.measure(*d.qubits,
                                    invert_mask=(True, False, False)))


def test_validate_circuit():
    d = generic_device(2)
    circuit1 = cirq.Circuit(device=d)
    circuit1.append(cirq.X(cirq.NamedQubit('q1')))
    circuit1.append(cirq.measure(cirq.NamedQubit('q1')))
    d.validate_circuit(circuit1)
    circuit1.append(cirq.CX(cirq.NamedQubit('q1'), cirq.NamedQubit('q0')))
    with pytest.raises(ValueError, match="Non-empty moment after measurement"):
        d.validate_circuit(circuit1)


def test_value_equal():
    dev = PasqalDevice(qubits=[cirq.NamedQubit('q1')])

    assert PasqalDevice(qubits=[cirq.NamedQubit('q1')]) == dev


def test_repr():
    assert repr(generic_device(1)) == ("pasqal.PasqalDevice("
                                       "qubits=[cirq.NamedQubit('q0')])")


def test_to_json():
    dev = PasqalDevice(qubits=[cirq.NamedQubit('q4')])
    d = dev._json_dict_()
    assert d == {"cirq_type": "PasqalDevice", "qubits": [cirq.NamedQubit('q4')]}

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
import sympy
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
    d = generic_device(2)
    op = (cirq.ops.CZ).on(*(d.qubit_list()))
    assert d.decompose_operation(op) == [op]

    op = op**sympy.Symbol('exp')
    with pytest.raises(TypeError, match="Don't know how to work with "):
        d.decompose_operation(op)

    # MeasurementGate is not a GateOperation
    with pytest.raises(TypeError):
        d.decompose_operation(cirq.ops.MeasurementGate(num_qubits=2))
    # It has to be made into one
    assert d.is_pasqal_device_op(
        cirq.ops.GateOperation(cirq.ops.MeasurementGate(2),
                               [cirq.NamedQubit('q0'),
                                cirq.NamedQubit('q1')]))


def test_is_pasqal_device():
    d = generic_device(2)

    with pytest.raises(ValueError, match="Got unknown operation"):
        d.is_pasqal_device_op(cirq.NamedQubit('q0'))

    op = (cirq.ops.CZ).on(*(d.qubit_list()))
    bad_op = cirq.ops.CNotPowGate(exponent=0.5)

    assert d.is_pasqal_device_op(op)
    assert d.is_pasqal_device_op(cirq.ops.X(cirq.NamedQubit('q0')))
    assert not d.is_pasqal_device_op(
        cirq.ops.CCX(cirq.NamedQubit('q0'), cirq.NamedQubit('q1'),
                     cirq.NamedQubit('q2'))**0.2)
    assert not d.is_pasqal_device_op(
        bad_op(cirq.NamedQubit('q0'), cirq.NamedQubit('q1')))
    op1 = cirq.ops.CNotPowGate(exponent=1.)
    assert d.is_pasqal_device_op(
        op1(cirq.NamedQubit('q0'), cirq.NamedQubit('q1')))

    op2 = (cirq.ops.H**sympy.Symbol('exp')).on(d.qubit_list()[0])
    assert not d.is_pasqal_device_op(op2)

    decomp = d.decompose_operation(op2)
    for op_ in decomp:
        assert d.is_pasqal_device_op(op_)


def test_decompose_operation():
    d = generic_device(3)
    for op in d.decompose_operation((cirq.CCZ**1.5).on(*(d.qubit_list()))):
        d.validate_operation(op)


def test_pasqal_converter():
    q = cirq.NamedQubit.range(2, prefix='q')
    g = cirq.TwoQubitGate()

    class FakeOperation(cirq.ops.GateOperation):

        def __init__(self, gate, qubits):
            self._gate = gate
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            return FakeOperation(self._gate, new_qubits)

    op = FakeOperation(g, q).with_qubits(*q)
    d = PasqalDevice(q)

    with pytest.raises(TypeError, match="Don't know how to work with"):
        d.decompose_operation(op)


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

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

from cirq.pasqal import PasqalDevice, PasqalVirtualDevice


def generic_device(num_qubits) -> PasqalDevice:
    return PasqalDevice(qubits=cirq.NamedQubit.range(num_qubits, prefix='q'))

def square_virtual_device(control_r, num_qubits) -> PasqalVirtualDevice:
    return PasqalVirtualDevice(control_radius=control_r,
                        qubits=cirq.pasqal.TwoDQubit.square(num_qubits))

def test_init():
    d = generic_device(3)
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')

    assert d.qubit_set() == {q0, q1, q2}
    assert d.qubit_list() == [q0, q1, q2]

def test_init_errors():
    line = cirq.devices.LineQubit.range(3)
    with pytest.raises(TypeError, match="Unsupported qubit type"):
        PasqalDevice(qubits=line)

    with pytest.raises(ValueError,
                       match="control_radius needs to be a non-negative float"):
        square_virtual_device(control_r=-1., num_qubits=2)

    with pytest.raises(ValueError,
                       match="control_radius cannot be larger than "
                       "5 times the minimal distance between qubits."):
        square_virtual_device(control_r=11., num_qubits=2)
    # with pytest.raises(ValueError, match='needs at least one qubit.'):
    #     generic_device(0)


def test_decompose_error():
    d = generic_device(2)
    op = (cirq.ops.CZ).on(*(d.qubit_list()))
    assert d.decompose_operation(op) == [op]

    # op = cirq.H.on(cirq.NamedQubit('q0'))
    # decomposition = d.decompose_operation(op)
    # assert len(decomposition) == 2
    # print(decomposition)
    # assert decomposition == [
    #     (cirq.Y**0.5).on(cirq.NamedQubit('q0')),
    #     cirq.XPowGate(exponent=1.0,
    #                   global_shift=-0.25).on(cirq.NamedQubit('q0'))
    # ]

    # MeasurementGate is not a GateOperation
    with pytest.raises(TypeError):
        d.decompose_operation(cirq.ops.MeasurementGate(num_qubits=2))
    # It has to be made into one
    assert d.is_pasqal_device_op(
        cirq.ops.GateOperation(cirq.ops.MeasurementGate(2),
                               [cirq.NamedQubit('q0'), cirq.NamedQubit('q1')]))

    assert d.is_pasqal_device_op(cirq.ops.X(cirq.NamedQubit('q0')))


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

    with pytest.raises(ValueError,
                       match="All qubits have to be measured at "
                       "once on a PasqalDevice."):
        circuit.append(cirq.measure(cirq.NamedQubit('q0')))

    with pytest.raises(NotImplementedError,
                       match="Measurements on Pasqal devices "
                       "don't support invert_mask."):
        circuit.append(cirq.measure(
            *d.qubits, invert_mask=(True, False, False)))

    d = square_virtual_device(control_r=1., num_qubits=3)

    with pytest.raises(ValueError, match="are too far away"):
        d.validate_operation(
            cirq.CZ.on(cirq.pasqal.TwoDQubit(0, 0),
                       cirq.pasqal.TwoDQubit(2, 2)))

def test_validate_moment():
    d = square_virtual_device(control_r=1., num_qubits=2)
    m = cirq.Moment([cirq.Z.on(cirq.pasqal.TwoDQubit(0, 0)),
            (cirq.X).on(cirq.pasqal.TwoDQubit(1, 1))])

    with pytest.raises(ValueError, match="Cannot do simultaneous gates"):
        d.validate_moment(m)

def test_minimal_distance():
    dev = square_virtual_device(control_r=1., num_qubits=1)

    with pytest.raises(ValueError,
                       match="Two qubits to compute a minimal distance."):
        dev.minimal_distance()

def test_distance():
    dev = square_virtual_device(control_r=1., num_qubits=2)

    with pytest.raises(ValueError,
                       match="Qubit not part of the device."):
        dev.distance(cirq.pasqal.TwoDQubit(0, 0), cirq.pasqal.TwoDQubit(2, 2))

def test_value_equal():
    dev = cirq.pasqal.PasqalDevice(qubits=[cirq.NamedQubit('q1')])

    assert cirq.pasqal.PasqalDevice(qubits=[cirq.NamedQubit('q1')]) == dev

    dev = cirq.pasqal.PasqalVirtualDevice(control_radius=1.,
                                          qubits=[cirq.pasqal.TwoDQubit(0, 0)])

    assert cirq.pasqal.PasqalVirtualDevice(control_radius=1.,
                                qubits=[cirq.pasqal.TwoDQubit(0, 0)]) == dev

def test_repr():
    assert repr(generic_device(1)) == ("pasqal.PasqalDevice("
                                       "qubits=[cirq.NamedQubit('q0')])")

    dev = cirq.pasqal.PasqalVirtualDevice(control_radius=1.,
                                          qubits=[cirq.pasqal.TwoDQubit(0, 0)])
    assert repr(dev) == ("pasqal.PasqalVirtualDevice("
                         "control_radius=1.0, "
                         "qubits=[pasqal.TwoDQubit(0, 0)])")

def test_to_json():
    dev = cirq.pasqal.PasqalDevice(qubits=[cirq.NamedQubit('q4')])
    d = dev._json_dict_()
    assert d == {
        "cirq_type": "PasqalDevice",
        "qubits": [cirq.NamedQubit('q4')]
    }
    vdev = cirq.pasqal.PasqalVirtualDevice(control_radius=2,
        qubits=[cirq.pasqal.TwoDQubit(0, 0)])
    d = vdev._json_dict_()
    assert d == {
        "cirq_type": "PasqalVirtualDevice",
        "control_radius": 2,
        "qubits": [cirq.pasqal.TwoDQubit(0, 0)]
    }

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
import numpy as np

import cirq

from cirq.pasqal import PasqalDevice, PasqalVirtualDevice
from cirq.pasqal import TwoDQubit, ThreeDQubit


def generic_device(num_qubits) -> PasqalDevice:
    return PasqalDevice(qubits=cirq.NamedQubit.range(num_qubits, prefix='q'))

def square_virtual_device(control_r, num_qubits) -> PasqalVirtualDevice:
    return PasqalVirtualDevice(control_radius=control_r,
                        qubits=TwoDQubit.square(num_qubits))

def test_init():
    d = generic_device(3)
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')

    assert d.qubit_set() == {q0, q1, q2}
    assert d.qubit_list() == [q0, q1, q2]
    assert d.supported_qubit_type == (cirq.NamedQubit,)

    d = square_virtual_device(control_r=1., num_qubits=1)

    assert d.qubit_set() == {TwoDQubit(0,0)}
    assert d.qubit_list() == [TwoDQubit(0,0)]
    assert d.control_radius == 1.
    assert d.supported_qubit_type == (ThreeDQubit, TwoDQubit, cirq.GridQubit,
                                        cirq.LineQubit,)

def test_init_errors():
    line = cirq.devices.LineQubit.range(3)
    with pytest.raises(TypeError, match="Unsupported qubit type"):
        PasqalDevice(qubits=line)

    with pytest.raises(TypeError, match="Unsupported qubit type"):
        PasqalVirtualDevice(control_radius=1.,
                            qubits=[cirq.NamedQubit('q0')])

    with pytest.raises(TypeError,
                       match="All qubits must be of same type."):
        PasqalVirtualDevice(control_radius=1.,
                            qubits=[TwoDQubit(0,0), cirq.GridQubit(1,0)])

    with pytest.raises(ValueError,
                       match="control_radius needs to be a non-negative float"):
        square_virtual_device(control_r=-1., num_qubits=2)

    with pytest.raises(ValueError,
                       match="control_radius cannot be larger than "
                       "5 times the minimal distance between qubits."):
        square_virtual_device(control_r=11., num_qubits=2)

def test_decompose_error():
    d = generic_device(2)
    op = (cirq.ops.CZ).on(*(d.qubit_list()))
    assert d.decompose_operation(op) == [op]

    # MeasurementGate is not a GateOperation
    with pytest.raises(TypeError):
        d.decompose_operation(cirq.ops.MeasurementGate(num_qubits=2))
    # It has to be made into one
    assert d.is_pasqal_device_op(
        cirq.ops.GateOperation(cirq.ops.MeasurementGate(2),
                               [cirq.NamedQubit('q0'), cirq.NamedQubit('q1')]))

    assert d.is_pasqal_device_op(cirq.ops.X(cirq.NamedQubit('q0')))

def test_decompose_operation():
    d = generic_device(3)
    op = (cirq.ops.CCZ).on(*(d.qubit_list()))


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
                       match="is not part of the device."):
        d.validate_operation(cirq.X.on(cirq.NamedQubit('q6')))

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
        d.validate_operation(cirq.CZ.on(TwoDQubit(0, 0), TwoDQubit(2, 2)))

def test_validate_moment():
    d = square_virtual_device(control_r=1., num_qubits=2)
    m = cirq.Moment([cirq.Z.on(TwoDQubit(0, 0)), (cirq.X).on(TwoDQubit(1, 1))])

    with pytest.raises(ValueError, match="Cannot do simultaneous gates"):
        d.validate_moment(m)

def test_minimal_distance():
    dev = square_virtual_device(control_r=1., num_qubits=1)

    with pytest.raises(ValueError,
                       match="There is no minimal distance for a "
                           "single-qubit."):
        dev.minimal_distance()

    dev = square_virtual_device(control_r=1., num_qubits=2)

    assert np.isclose(dev.minimal_distance(), 1.)

def test_distance():
    dev = square_virtual_device(control_r=1., num_qubits=2)

    with pytest.raises(ValueError,
                       match="Qubit not part of the device."):
        dev.distance(TwoDQubit(0, 0), TwoDQubit(2, 2))

    assert np.isclose(dev.distance(TwoDQubit(0, 0), TwoDQubit(1, 0)), 1.)

    dev = PasqalVirtualDevice(control_radius=1.,
                              qubits=[cirq.LineQubit(0), cirq.LineQubit(1)])

    assert np.isclose(dev.distance(cirq.LineQubit(0), cirq.LineQubit(1)), 1.)

    dev = PasqalVirtualDevice(control_radius=1.,
        qubits=[cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)])

    assert np.isclose(dev.distance(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)), 1.)

def test_value_equal():
    dev = PasqalDevice(qubits=[cirq.NamedQubit('q1')])

    assert PasqalDevice(qubits=[cirq.NamedQubit('q1')]) == dev

    dev = PasqalVirtualDevice(control_radius=1., qubits=[TwoDQubit(0, 0)])

    assert PasqalVirtualDevice(control_radius=1.,
                               qubits=[TwoDQubit(0, 0)]) == dev

def test_repr():
    assert repr(generic_device(1)) == ("pasqal.PasqalDevice("
                                       "qubits=[cirq.NamedQubit('q0')])")

    dev = PasqalVirtualDevice(control_radius=1., qubits=[TwoDQubit(0, 0)])
    assert repr(dev) == ("pasqal.PasqalVirtualDevice("
                         "control_radius=1.0, "
                         "qubits=[pasqal.TwoDQubit(0, 0)])")

def test_to_json():
    dev = PasqalDevice(qubits=[cirq.NamedQubit('q4')])
    d = dev._json_dict_()
    assert d == {
        "cirq_type": "PasqalDevice",
        "qubits": [cirq.NamedQubit('q4')]
    }
    vdev = PasqalVirtualDevice(control_radius=2,
        qubits=[TwoDQubit(0, 0)])
    d = vdev._json_dict_()
    assert d == {
        "cirq_type": "PasqalVirtualDevice",
        "control_radius": 2,
        "qubits": [cirq.pasqal.TwoDQubit(0, 0)]
    }

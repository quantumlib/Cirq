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


def test_init_errors():
    line = cirq.devices.LineQubit.range(3)
    with pytest.raises(TypeError, match="Unsupported qubit type"):
        PasqalDevice(qubits=line)


def test_decompose_error():
    d = generic_device(2)
    op = (cirq.ops.CZ).on(*(d.qubit_list()))
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
        d.decompose_operation(cirq.ops.MeasurementGate(num_qubits=1))
    # It has to be made into one
    assert PasqalDevice.is_pasqal_device_op(
        cirq.ops.GateOperation(cirq.ops.MeasurementGate(1),
                               [cirq.NamedQubit('q0')]))

    assert PasqalDevice.is_pasqal_device_op(cirq.ops.X(cirq.NamedQubit('q0')))


# def test_validate_operation_errors():
#     d = cubic_device(3, 3, 3)
#     qlist = d.qubit_list()
#     too_many_qubits_op = cirq.ops.X.controlled(len(qlist) - 1)
#     too_many_qubits_op = cirq.ops.GateOperation(too_many_qubits_op, qlist)
#
#     with pytest.raises(ValueError,
#                        match="Too many qubits acted on in parallel by"):
#         d.validate_operation(too_many_qubits_op)
#
#     with pytest.raises(ValueError, match="Unsupported operation"):
#         d.validate_operation(ThreeDGridQubit(0, 0, 0))
#
#     with pytest.raises(ValueError, match="cirq.H is not a supported gate"):
#         d.validate_operation(cirq.ops.H.on(ThreeDGridQubit(0, 0, 0)))
#
#     with pytest.raises(ValueError,
#                        match="is not a 3D grid qubit for gate cirq.X"):
#         d.validate_operation(cirq.X.on(cirq.LineQubit(0)))
#
#     with pytest.raises(ValueError, match="are too far away"):
#         d.validate_operation(
#             cirq.CZ.on(ThreeDGridQubit(0, 0, 0), ThreeDGridQubit(3, 3, 3)))
#
#     with pytest.raises(ValueError, match="Too many Z gates in parallel"):
#         d.validate_operation(cirq.ParallelGateOperation(cirq.ops.Z, d.qubits))
#
#     with pytest.raises(ValueError, match="Bad number of X/Y gates in parallel"):
#         d.validate_operation(
#             cirq.ParallelGateOperation(cirq.ops.X,
#                                        d.qubit_list()[1:]))
#
#     assert d.validate_operation(
#         cirq.ops.GateOperation(cirq.ops.MeasurementGate(1),
#                                [ThreeDGridQubit(0, 0, 0)])) is None
#
#
# def test_qubit_set():
#     assert cubic_device(2, 2,
#                         2).qubit_set() == set(ThreeDGridQubit.cube(2, 0, 0, 0))
#
#
# def test_distance():
#     d = cubic_device(2, 2, 1)
#     assert d.distance(ThreeDGridQubit(0, 0, 0), ThreeDGridQubit(1, 0, 0)) == 1
#
#     with pytest.raises(TypeError):
#         _ = d.distance(ThreeDGridQubit(0, 0, 0), cirq.devices.LineQubit(1))
#
#     with pytest.raises(TypeError):
#         _ = d.distance(cirq.devices.LineQubit(1), ThreeDGridQubit(0, 0, 0))
#
#
# def test_value_equal():
#     dev = cirq.pasqal.PasqalDevice(control_radius=5,
#                                    qubits=[ThreeDGridQubit(1, 1, 1)])
#
#     assert cirq.pasqal.PasqalDevice(control_radius=5,
#                                     qubits=[ThreeDGridQubit(1, 1, 1)]) == dev
#
#
# def test_repr():
#     assert repr(cubic_device(
#         1, 1, 1)) == ("pasqal.PasqalDevice("
#                       "control_radius=1.5,"
#                       " qubits=[pasqal.ThreeDGridQubit(0, 0, 0)])")
#
#
# def test_to_json():
#     dev = cirq.pasqal.PasqalDevice(
#         control_radius=5, qubits=[cirq.pasqal.ThreeDGridQubit(1, 1, 1)])
#     d = dev._json_dict_()
#     assert d == {
#         "cirq_type": "PasqalDevice",
#         "control_radius": 5,
#         "qubits": [cirq.pasqal.ThreeDGridQubit(1, 1, 1)]
#     }

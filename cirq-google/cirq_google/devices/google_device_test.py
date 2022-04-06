# Copyright 2022 The Cirq Developers
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

import unittest.mock as mock

import pytest

import cirq
import cirq_google
from cirq_google.api import v2


GRID_HEIGHT = 5


def _create_device_spec_with_horizontal_couplings():
    # Qubit layout:
    #   x -- x
    #   x -- x
    #   x -- x
    #   x -- x
    #   x -- x

    grid_qubits = [cirq.GridQubit(i, j) for i in range(GRID_HEIGHT) for j in range(2)]
    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([v2.qubit_to_proto_id(q) for q in grid_qubits])
    grid_targets = spec.valid_targets.add()
    grid_targets.name = '2_qubit_targets'
    grid_targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    for row in range(GRID_HEIGHT):
        new_target = grid_targets.targets.add()
        new_target.ids.extend([v2.qubit_to_proto_id(cirq.GridQubit(row, j)) for j in range(2)])
    gate = spec.valid_gates.add()
    gate.syc.SetInParent()
    gate.gate_duration_picos = 12000
    gate.valid_targets.extend(['2_qubit_targets'])

    return grid_qubits, spec


def _create_device_spec_with_all_couplings():
    # Qubit layout:
    #   x -- x
    #   |    |
    #   x -- x
    #   |    |
    #   x -- x
    #   |    |
    #   x -- x
    #   |    |
    #   x -- x

    grid_qubits, spec = _create_device_spec_with_horizontal_couplings()
    for row in range(GRID_HEIGHT - 1):
        for col in range(2):
            new_target = spec.valid_targets[0].targets.add()
            new_target.ids.extend(
                [
                    v2.qubit_to_proto_id(cirq.GridQubit(row, col)),
                    v2.qubit_to_proto_id(cirq.GridQubit(row + 1, col)),
                ]
            )
    return grid_qubits, spec


def _create_device_spec_with_qubit_pair_self_loops() -> v2.device_pb2.DeviceSpecification:
    q_proto_id = v2.qubit_to_proto_id(cirq.NamedQubit('q'))

    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_id])
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = targets.targets.add()
    new_target.ids.extend([q_proto_id, q_proto_id])

    return spec


def _create_device_spec_with_invalid_qubit_in_qubit_pair() -> v2.device_pb2.DeviceSpecification:
    q_proto_ids = [v2.qubit_to_proto_id(cirq.GridQubit(0, i)) for i in range(2)]

    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_ids[0]])
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = targets.targets.add()
    new_target.ids.extend([q_proto_ids[0], q_proto_ids[1]])

    return spec


def test_google_device_from_proto_and_validation():
    grid_qubits, spec = _create_device_spec_with_horizontal_couplings()

    device = cirq_google.GoogleDevice.from_proto(spec)

    assert len(device.metadata.qubit_set) == len(grid_qubits)
    assert device.metadata.qubit_set == frozenset(grid_qubits)
    assert all(
        (cirq.GridQubit(row, 0), cirq.GridQubit(row, 1)) in device.metadata.qubit_pairs
        for row in range(GRID_HEIGHT)
    )


def test_google_device_validate_operations_positive():
    grid_qubits, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GoogleDevice.from_proto(spec)

    for q in grid_qubits:
        device.validate_operation(cirq.X(q))

    # horizontal qubit pairs
    for i in range(GRID_HEIGHT):
        device.validate_operation(cirq.CZ(grid_qubits[2 * i], grid_qubits[2 * i + 1]))

    # TODO(#5050) verify validate_operations gateset support


def test_google_device_validate_operations_negative():
    grid_qubits, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GoogleDevice.from_proto(spec)

    q = cirq.GridQubit(10, 10)
    with pytest.raises(ValueError, match='Qubit not on device'):
        device.validate_operation(cirq.X(q))

    # vertical qubit pair
    q00, q10 = grid_qubits[0], grid_qubits[2]  # (0, 0), (1, 0)
    with pytest.raises(ValueError, match='Qubit pair is not valid'):
        device.validate_operation(cirq.CZ(q00, q10))

    # TODO(#5050) verify validate_operations gateset errors


@pytest.mark.parametrize(
    'spec',
    [
        # TODO(#5050) implement once gateset support is implemented
        # _create_device_spec_with_missing_gate_durations(),
        _create_device_spec_with_qubit_pair_self_loops(),
        _create_device_spec_with_invalid_qubit_in_qubit_pair(),
    ],
)
def test_google_device_invalid_device_spec(spec):
    with pytest.raises(ValueError, match='DeviceSpecification is invalid'):
        cirq_google.GoogleDevice.from_proto(spec)


def test_google_device_repr_json():
    _, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GoogleDevice.from_proto(spec)

    assert eval(repr(device)) == device
    assert cirq.read_json(json_text=cirq.to_json(device)) == device


def test_google_device_str_grid_qubits():
    _, spec = _create_device_spec_with_all_couplings()
    device = cirq_google.GoogleDevice.from_proto(spec)

    assert (
        str(device)
        == """\
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)
│        │
│        │
(2, 0)───(2, 1)
│        │
│        │
(3, 0)───(3, 1)
│        │
│        │
(4, 0)───(4, 1)"""
    )


@pytest.mark.parametrize('cycle,func', [(False, str), (True, repr)])
def test_google_device_repr_pretty(cycle, func):
    _, spec = _create_device_spec_with_all_couplings()
    device = cirq_google.GoogleDevice.from_proto(spec)
    printer = mock.Mock()
    device._repr_pretty_(printer, cycle)
    printer.text.assert_called_once_with(func(device))


def test_serializable_device_str_named_qubits():
    q_proto_id = v2.qubit_to_proto_id(cirq.NamedQubit('q'))
    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_id])
    device = cirq_google.GoogleDevice.from_proto(spec)
    assert device.__class__.__name__ in str(device)

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
    for row in range(int(GRID_HEIGHT / 2)):
        new_target = grid_targets.targets.add()
        new_target.ids.extend([v2.qubit_to_proto_id(cirq.GridQubit(row, j)) for j in range(2)])
    for row in range(int(GRID_HEIGHT / 2), GRID_HEIGHT):
        # Flip the qubit pair order for the second half of qubits
        # to verify GridDevice properly handles pair symmetry.
        new_target = grid_targets.targets.add()
        new_target.ids.extend([v2.qubit_to_proto_id(cirq.GridQubit(row, 1 - j)) for j in range(2)])
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


def _create_device_spec_duplicate_qubit() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification with a qubit name that does not conform to '<int>_<int>'."""
    q_proto_id = v2.qubit_to_proto_id(cirq.GridQubit(0, 0))

    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_id, q_proto_id])

    return spec


def _create_device_spec_invalid_qubit_name() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification with a qubit name that does not conform to '<int>_<int>'."""
    q_proto_id = v2.qubit_to_proto_id(cirq.NamedQubit('q0_0'))

    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_id])

    return spec


def _create_device_spec_qubit_pair_self_loops() -> v2.device_pb2.DeviceSpecification:
    """Creates an invalid DeviceSpecification with a qubit pair ('0_0', '0_0')."""

    q_proto_id = v2.qubit_to_proto_id(cirq.GridQubit(0, 0))

    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_id])
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = targets.targets.add()
    new_target.ids.extend([q_proto_id, q_proto_id])

    return spec


def _create_device_spec_invalid_qubit_in_qubit_pair() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification where qubit '0_1' is in a pair but not in valid_qubits."""

    q_proto_ids = [v2.qubit_to_proto_id(cirq.GridQubit(0, i)) for i in range(2)]

    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([q_proto_ids[0]])
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = targets.targets.add()
    new_target.ids.extend([q_proto_ids[0], q_proto_ids[1]])

    return spec


def _create_device_spec_invalid_subset_permutation_target() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification where a SUBSET_PERMUTATION target contains 2 qubits."""

    q_proto_ids = [v2.qubit_to_proto_id(cirq.GridQubit(0, i)) for i in range(2)]

    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend(q_proto_ids)
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.SUBSET_PERMUTATION
    new_target = targets.targets.add()
    new_target.ids.extend(q_proto_ids)  # should only have 1 qubit instead

    return spec


def test_grid_device_from_proto():
    grid_qubits, spec = _create_device_spec_with_horizontal_couplings()

    device = cirq_google.GridDevice.from_proto(spec)

    assert len(device.metadata.qubit_set) == len(grid_qubits)
    assert device.metadata.qubit_set == frozenset(grid_qubits)
    assert all(
        frozenset((cirq.GridQubit(row, 0), cirq.GridQubit(row, 1))) in device.metadata.qubit_pairs
        for row in range(GRID_HEIGHT)
    )


def test_grid_device_validate_operations_positive():
    grid_qubits, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)

    for q in grid_qubits:
        device.validate_operation(cirq.X(q))

    # horizontal qubit pairs
    for i in range(GRID_HEIGHT):
        device.validate_operation(cirq.CZ(grid_qubits[2 * i], grid_qubits[2 * i + 1]))

    # TODO(#5050) verify validate_operations gateset support


def test_grid_device_validate_operations_negative():
    grid_qubits, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)

    q = cirq.GridQubit(10, 10)
    with pytest.raises(ValueError, match='Qubit not on device'):
        device.validate_operation(cirq.X(q))

    # vertical qubit pair
    q00, q10 = grid_qubits[0], grid_qubits[2]  # (0, 0), (1, 0)
    with pytest.raises(ValueError, match='Qubit pair is not valid'):
        device.validate_operation(cirq.CZ(q00, q10))

    # TODO(#5050) verify validate_operations gateset errors


@pytest.mark.parametrize(
    'spec, error_match',
    [
        (_create_device_spec_duplicate_qubit(), 'Invalid DeviceSpecification: .*duplicate qubit'),
        (
            _create_device_spec_invalid_qubit_name(),
            'Invalid DeviceSpecification: .*not in the GridQubit form',
        ),
        (
            _create_device_spec_invalid_qubit_in_qubit_pair(),
            'Invalid DeviceSpecification: .*which is not in valid_qubits',
        ),
        (
            _create_device_spec_qubit_pair_self_loops(),
            'Invalid DeviceSpecification: .*contains repeated qubits',
        ),
        (
            _create_device_spec_invalid_subset_permutation_target(),
            'Invalid DeviceSpecification: .*does not have exactly 1 qubit',
        ),
    ],
)
def test_grid_device_invalid_device_specification(spec, error_match):
    with pytest.raises(ValueError, match=error_match):
        cirq_google.GridDevice.from_proto(spec)


def test_grid_device_repr_json():
    _, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)

    assert eval(repr(device)) == device
    assert cirq.read_json(json_text=cirq.to_json(device)) == device


def test_grid_device_str_grid_qubits():
    _, spec = _create_device_spec_with_all_couplings()
    device = cirq_google.GridDevice.from_proto(spec)

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
def test_grid_device_repr_pretty(cycle, func):
    _, spec = _create_device_spec_with_all_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    printer = mock.Mock()
    device._repr_pretty_(printer, cycle)
    printer.text.assert_called_once_with(func(device))

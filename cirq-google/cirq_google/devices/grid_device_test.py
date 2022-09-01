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

from dataclasses import dataclass
from typing import Dict, List, Tuple

import unittest.mock as mock
import pytest
from google.protobuf import text_format

import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device


GRID_HEIGHT = 5


@dataclass
class _DeviceInfo:
    """Dataclass for device information relevant to GridDevice tests."""

    grid_qubits: List[cirq.GridQubit]
    qubit_pairs: List[Tuple[cirq.GridQubit, cirq.GridQubit]]
    expected_gateset: cirq.Gateset
    expected_gate_durations: Dict[cirq.GateFamily, cirq.Duration]
    expected_target_gatesets: Tuple[cirq.CompilationTargetGateset, ...]


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

    qubit_pairs = []
    grid_targets = spec.valid_targets.add()
    grid_targets.name = '2_qubit_targets'
    grid_targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    for row in range(int(GRID_HEIGHT / 2)):
        qubit_pairs.append((cirq.GridQubit(row, 0), cirq.GridQubit(row, 1)))
    for row in range(int(GRID_HEIGHT / 2), GRID_HEIGHT):
        # Flip the qubit pair order for the second half of qubits
        # to verify GridDevice properly handles pair symmetry.
        qubit_pairs.append((cirq.GridQubit(row, 1), cirq.GridQubit(row, 0)))
    for pair in qubit_pairs:
        new_target = grid_targets.targets.add()
        new_target.ids.extend([v2.qubit_to_proto_id(q) for q in pair])

    gate_names = [
        'syc',
        'sqrt_iswap',
        'sqrt_iswap_inv',
        'cz',
        'phased_xz',
        'virtual_zpow',
        'physical_zpow',
        'coupler_pulse',
        'meas',
        'wait',
    ]
    gate_durations = [(n, i * 1000) for i, n in enumerate(gate_names)]
    for gate_name, duration in sorted(gate_durations):
        gate = spec.valid_gates.add()
        getattr(gate, gate_name).SetInParent()
        gate.gate_duration_picos = duration

    expected_gateset = cirq.Gateset(
        cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]),
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]),
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]),
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]),
        cirq.ops.phased_x_z_gate.PhasedXZGate,
        cirq.ops.common_gates.XPowGate,
        cirq.ops.common_gates.YPowGate,
        cirq.ops.phased_x_gate.PhasedXPowGate,
        cirq.GateFamily(
            cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]
        ),
        cirq.GateFamily(
            cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]
        ),
        cirq_google.experimental.ops.coupler_pulse.CouplerPulse,
        cirq.ops.measurement_gate.MeasurementGate,
        cirq.ops.wait_gate.WaitGate,
    )

    base_duration = cirq.Duration(picos=1_000)
    expected_gate_durations = {
        cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]): base_duration * 0,
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]): base_duration * 1,
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]): base_duration * 2,
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]): base_duration * 3,
        cirq.GateFamily(cirq.ops.phased_x_z_gate.PhasedXZGate): base_duration * 4,
        cirq.GateFamily(cirq.ops.common_gates.XPowGate): base_duration * 4,
        cirq.GateFamily(cirq.ops.common_gates.YPowGate): base_duration * 4,
        cirq.GateFamily(cirq.ops.phased_x_gate.PhasedXPowGate): base_duration * 4,
        cirq.GateFamily(
            cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]
        ): base_duration
        * 5,
        cirq.GateFamily(
            cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]
        ): base_duration
        * 6,
        cirq.GateFamily(cirq_google.experimental.ops.coupler_pulse.CouplerPulse): base_duration * 7,
        cirq.GateFamily(cirq.ops.measurement_gate.MeasurementGate): base_duration * 8,
        cirq.GateFamily(cirq.ops.wait_gate.WaitGate): base_duration * 9,
    }

    expected_target_gatesets = (
        cirq_google.GoogleCZTargetGateset(
            additional_gates=[
                cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]),
                cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]),
                cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]),
                cirq.ops.common_gates.XPowGate,
                cirq.ops.common_gates.YPowGate,
                cirq.ops.phased_x_gate.PhasedXPowGate,
                cirq.GateFamily(
                    cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]
                ),
                cirq.GateFamily(
                    cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]
                ),
                cirq_google.experimental.ops.coupler_pulse.CouplerPulse,
                cirq.ops.wait_gate.WaitGate,
            ]
        ),
        cirq_google.SycamoreTargetGateset(),
        cirq.SqrtIswapTargetGateset(
            additional_gates=[
                cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]),
                cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]),
                cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]),
                cirq.ops.common_gates.XPowGate,
                cirq.ops.common_gates.YPowGate,
                cirq.ops.phased_x_gate.PhasedXPowGate,
                cirq.GateFamily(
                    cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]
                ),
                cirq.GateFamily(
                    cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]
                ),
                cirq_google.experimental.ops.coupler_pulse.CouplerPulse,
                cirq.ops.wait_gate.WaitGate,
            ]
        ),
    )

    return (
        _DeviceInfo(
            grid_qubits,
            qubit_pairs,
            expected_gateset,
            expected_gate_durations,
            expected_target_gatesets,
        ),
        spec,
    )


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

    _, spec = _create_device_spec_with_horizontal_couplings()
    for row in range(GRID_HEIGHT - 1):
        for col in range(2):
            new_target = spec.valid_targets[0].targets.add()
            new_target.ids.extend(
                [
                    v2.qubit_to_proto_id(cirq.GridQubit(row, col)),
                    v2.qubit_to_proto_id(cirq.GridQubit(row + 1, col)),
                ]
            )
    return spec


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


def _create_device_spec_unexpected_asymmetric_target() -> v2.device_pb2.DeviceSpecification:
    """Creates a DeviceSpecification containing an ASYMMETRIC target set."""

    spec = v2.device_pb2.DeviceSpecification()
    targets = spec.valid_targets.add()
    targets.name = 'test_targets'
    targets.target_ordering = v2.device_pb2.TargetSet.ASYMMETRIC

    return spec


def test_grid_device_from_proto():
    device_info, spec = _create_device_spec_with_horizontal_couplings()

    device = cirq_google.GridDevice.from_proto(spec)

    assert len(device.metadata.qubit_set) == len(device_info.grid_qubits)
    assert device.metadata.qubit_set == frozenset(device_info.grid_qubits)
    assert all(frozenset(pair) in device.metadata.qubit_pairs for pair in device_info.qubit_pairs)
    assert device.metadata.gateset == device_info.expected_gateset
    assert device.metadata.gate_durations == device_info.expected_gate_durations
    assert (
        tuple(device.metadata.compilation_target_gatesets) == device_info.expected_target_gatesets
    )


def test_grid_device_validate_operations_positive():
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    # Gates that can be applied to any subset of valid qubits
    variadic_gates = [cirq.measure, cirq.WaitGate(cirq.Duration(nanos=1), num_qubits=2)]

    for q in device_info.grid_qubits:
        device.validate_operation(cirq.X(q))
        device.validate_operation(cirq.measure(q))

    # horizontal qubit pairs
    for i in range(GRID_HEIGHT):
        device.validate_operation(
            cirq.CZ(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * i + 1])
        )
        for gate in variadic_gates:
            device.validate_operation(
                gate(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * i + 1])
            )


@pytest.mark.parametrize(
    'gate_func',
    [
        lambda _: cirq.measure,
        lambda num_qubits: cirq.WaitGate(cirq.Duration(nanos=1), num_qubits=num_qubits),
    ],
)
def test_grid_device_validate_operations_variadic_gates_positive(gate_func):
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)

    # Single qubit operations
    for q in device_info.grid_qubits:
        device.validate_operation(gate_func(1)(q))

    # horizontal qubit pairs (coupled)
    for i in range(GRID_HEIGHT):
        device.validate_operation(
            gate_func(2)(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * i + 1])
        )

    # Variadic gates across vertical qubit pairs (uncoupled pairs) should succeed.
    for i in range(GRID_HEIGHT - 1):
        device.validate_operation(
            gate_func(2)(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * (i + 1)])
        )
        device.validate_operation(
            gate_func(2)(
                device_info.grid_qubits[2 * i + 1], device_info.grid_qubits[2 * (i + 1) + 1]
            )
        )

    # 3-qubit measurements
    for i in range(GRID_HEIGHT - 2):
        device.validate_operation(
            gate_func(3)(
                device_info.grid_qubits[2 * i],
                device_info.grid_qubits[2 * (i + 1)],
                device_info.grid_qubits[2 * (i + 2)],
            )
        )
        device.validate_operation(
            gate_func(3)(
                device_info.grid_qubits[2 * i + 1],
                device_info.grid_qubits[2 * (i + 1) + 1],
                device_info.grid_qubits[2 * (i + 2) + 1],
            )
        )
    # All-qubit measurement
    device.validate_operation(gate_func(len(device_info.grid_qubits))(*device_info.grid_qubits))


def test_grid_device_validate_operations_negative():
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)

    bad_qubit = cirq.GridQubit(10, 10)
    with pytest.raises(ValueError, match='Qubit not on device'):
        device.validate_operation(cirq.X(bad_qubit))

    # vertical qubit pair
    q00, q10 = device_info.grid_qubits[0], device_info.grid_qubits[2]  # (0, 0), (1, 0)
    with pytest.raises(ValueError, match='Qubit pair is not valid'):
        device.validate_operation(cirq.CZ(q00, q10))

    with pytest.raises(ValueError, match='gate which is not supported'):
        device.validate_operation(cirq.H(device_info.grid_qubits[0]))


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
            _create_device_spec_unexpected_asymmetric_target(),
            'Invalid DeviceSpecification: .*cannot be ASYMMETRIC',
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
    spec = _create_device_spec_with_all_couplings()
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
    spec = _create_device_spec_with_all_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    printer = mock.Mock()
    device._repr_pretty_(printer, cycle)
    printer.text.assert_called_once_with(func(device))


def test_to_proto():
    device_info, expected_spec = _create_device_spec_with_horizontal_couplings()

    # The set of gates in gate_durations are consistent with what's generated in
    # _create_device_spec_with_horizontal_couplings()
    base_duration = cirq.Duration(picos=1_000)
    gate_durations = {
        cirq.GateFamily(cirq_google.SYC): base_duration * 0,
        cirq.GateFamily(cirq.SQRT_ISWAP): base_duration * 1,
        cirq.GateFamily(cirq.SQRT_ISWAP_INV): base_duration * 2,
        cirq.GateFamily(cirq.CZ): base_duration * 3,
        cirq.GateFamily(cirq.ops.phased_x_z_gate.PhasedXZGate): base_duration * 4,
        cirq.GateFamily(
            cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]
        ): base_duration
        * 5,
        cirq.GateFamily(
            cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]
        ): base_duration
        * 6,
        cirq.GateFamily(cirq_google.experimental.ops.coupler_pulse.CouplerPulse): base_duration * 7,
        cirq.GateFamily(cirq.ops.measurement_gate.MeasurementGate): base_duration * 8,
        cirq.GateFamily(cirq.ops.wait_gate.WaitGate): base_duration * 9,
    }

    spec = grid_device._create_device_specification_proto(
        qubits=device_info.grid_qubits,
        pairs=device_info.qubit_pairs,
        gateset=cirq.Gateset(*gate_durations.keys()),
        gate_durations=gate_durations,
    )

    assert text_format.MessageToString(spec) == text_format.MessageToString(expected_spec)


@pytest.mark.parametrize(
    'error_match, qubits, qubit_pairs, gateset, gate_durations',
    [
        (
            'Gate durations contain keys which are not part of the gateset',
            [cirq.GridQubit(0, 0)],
            [],
            cirq.Gateset(cirq.CZ),
            {cirq.GateFamily(cirq.SQRT_ISWAP): 1_000},
        ),
        ('not in the GridQubit form', [cirq.NamedQubit('q0_0')], [], cirq.Gateset(), None),
        (
            'valid_targets contain .* which is not in valid_qubits',
            [cirq.GridQubit(0, 0)],
            [(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))],
            cirq.Gateset(),
            None,
        ),
        (
            'has a target which contains repeated qubits',
            [cirq.GridQubit(0, 0)],
            [(cirq.GridQubit(0, 0), cirq.GridQubit(0, 0))],
            cirq.Gateset(),
            None,
        ),
        ('Unrecognized gate', [cirq.GridQubit(0, 0)], [], cirq.Gateset(cirq.H), None),
    ],
)
def test_to_proto_invalid_input(error_match, qubits, qubit_pairs, gateset, gate_durations):
    with pytest.raises(ValueError, match=error_match):
        grid_device._create_device_specification_proto(
            qubits=qubits, pairs=qubit_pairs, gateset=gateset, gate_durations=gate_durations
        )


def test_to_proto_empty():
    spec = grid_device._create_device_specification_proto(
        # Qubits are always expected to be set
        qubits=[cirq.GridQubit(0, i) for i in range(5)],
        pairs=[],
        gateset=cirq.Gateset(),
        gate_durations=None,
    )
    device = cirq_google.GridDevice.from_proto(spec)

    assert len(device.metadata.qubit_set) == 5
    assert len(device.metadata.qubit_pairs) == 0
    assert device.metadata.gateset == cirq.Gateset()
    assert device.metadata.gate_durations is None


def test_to_proto_fsim_gate_family():
    """Verifies that FSimGateFamilies are serialized correctly."""

    gateset = cirq.Gateset(
        cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]),
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]),
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]),
        cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]),
    )

    spec = grid_device._create_device_specification_proto(
        qubits=[cirq.GridQubit(0, 0)], pairs=(), gateset=gateset
    )

    assert any(gate_spec.HasField('syc') for gate_spec in spec.valid_gates)
    assert any(gate_spec.HasField('sqrt_iswap') for gate_spec in spec.valid_gates)
    assert any(gate_spec.HasField('sqrt_iswap_inv') for gate_spec in spec.valid_gates)
    assert any(gate_spec.HasField('cz') for gate_spec in spec.valid_gates)

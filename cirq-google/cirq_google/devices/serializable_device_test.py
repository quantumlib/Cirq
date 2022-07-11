# Copyright 2019 The Cirq Developers
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
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.api.v2.device_pb2 as device_pb2


def _just_cz():
    # Deprecations: cirq_google.SerializableGateSet, cirq_google.GateOpSerializer, and
    # cirq_google.GateOpDeserializer
    with cirq.testing.assert_deprecated(
        'SerializableGateSet', 'CircuitSerializer', deadline='v0.16', count=None
    ):
        return cg.SerializableGateSet(
            gate_set_name='cz_gate_set',
            serializers=[
                cg.GateOpSerializer(gate_type=cirq.CZPowGate, serialized_gate_id='cz', args=[])
            ],
            deserializers=[
                cg.GateOpDeserializer(
                    serialized_gate_id='cz', gate_constructor=cirq.CZPowGate, args=[]
                )
            ],
        )


def _just_meas():
    # Deprecations: cirq_google.SerializableGateSet, cirq_google.GateOpSerializer, and
    # cirq_google.GateOpDeserializer
    with cirq.testing.assert_deprecated(
        'SerializableGateSet', 'CircuitSerializer', deadline='v0.16', count=None
    ):
        return cg.SerializableGateSet(
            gate_set_name='meas_gate_set',
            serializers=[
                cg.GateOpSerializer(
                    gate_type=cirq.MeasurementGate, serialized_gate_id='meas', args=[]
                )
            ],
            deserializers=[
                cg.GateOpDeserializer(
                    serialized_gate_id='meas', gate_constructor=cirq.MeasurementGate, args=[]
                )
            ],
        )


@pytest.mark.parametrize('cycle,func', [(False, str), (True, repr)])
def test_repr_pretty(cycle, func):
    device = cg.Sycamore
    printer = mock.Mock()
    device._repr_pretty_(printer, cycle)
    printer.text.assert_called_once_with(func(device))


def test_gate_definition_equality():
    def1 = cg.devices.serializable_device._GateDefinition(
        duration=cirq.Duration(picos=4),
        target_set={(cirq.GridQubit(1, 1),)},
        number_of_qubits=1,
        is_permutation=False,
    )
    def1c = cg.devices.serializable_device._GateDefinition(
        duration=cirq.Duration(picos=4),
        target_set={(cirq.GridQubit(1, 1),)},
        number_of_qubits=1,
        is_permutation=False,
    )
    def2 = cg.devices.serializable_device._GateDefinition(
        duration=cirq.Duration(picos=5),
        target_set={(cirq.GridQubit(1, 1),)},
        number_of_qubits=1,
        is_permutation=False,
    )
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(def1, def1c)
    eq.add_equality_group(def2)

    # Wrong type, tests NotImplemented functionality
    eq.add_equality_group(cirq.X)


def test_asymmetric_gate():
    spec = device_pb2.DeviceSpecification()
    for row in range(5):
        for col in range(2):
            spec.valid_qubits.extend([v2.qubit_to_proto_id(cirq.GridQubit(row, col))])
    grid_targets = spec.valid_targets.add()
    grid_targets.name = 'left_to_right'
    grid_targets.target_ordering = device_pb2.TargetSet.ASYMMETRIC
    for row in range(5):
        new_target = grid_targets.targets.add()
        new_target.ids.extend(
            [
                v2.qubit_to_proto_id(cirq.GridQubit(row, 0)),
                v2.qubit_to_proto_id(cirq.GridQubit(row, 1)),
            ]
        )

    gs_proto = spec.valid_gate_sets.add()
    gs_proto.name = 'cz_left_to_right_only'

    gate = gs_proto.valid_gates.add()
    gate.id = 'cz'
    gate.valid_targets.extend(['left_to_right'])

    cz_gateset = _just_cz()
    with cirq.testing.assert_deprecated('Use cirq_google.GridDevice', deadline='v0.16', count=2):
        dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[cz_gateset])

        for row in range(5):
            dev.validate_operation(cirq.CZ(cirq.GridQubit(row, 0), cirq.GridQubit(row, 1)))
            with pytest.raises(ValueError):
                dev.validate_operation(cirq.CZ(cirq.GridQubit(row, 1), cirq.GridQubit(row, 0)))


def test_unconstrained_gate():
    spec = device_pb2.DeviceSpecification()
    for row in range(5):
        for col in range(5):
            spec.valid_qubits.extend([v2.qubit_to_proto_id(cirq.GridQubit(row, col))])
    grid_targets = spec.valid_targets.add()
    grid_targets.name = '2_qubit_anywhere'
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    gs_proto = spec.valid_gate_sets.add()
    gs_proto.name = 'cz_free_for_all'

    gate = gs_proto.valid_gates.add()
    gate.id = 'cz'
    gate.valid_targets.extend(['2_qubit_anywhere'])

    cz_gateset = _just_cz()
    with cirq.testing.assert_deprecated('Use cirq_google.GridDevice', deadline='v0.16', count=2):
        dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[cz_gateset])

        valid_qubit1 = cirq.GridQubit(4, 4)
        for row in range(4):
            for col in range(4):
                valid_qubit2 = cirq.GridQubit(row, col)
                dev.validate_operation(cirq.CZ(valid_qubit1, valid_qubit2))


def test_number_of_qubits_cz():
    spec = device_pb2.DeviceSpecification()
    spec.valid_qubits.extend(
        [v2.qubit_to_proto_id(cirq.GridQubit(0, 0)), v2.qubit_to_proto_id(cirq.GridQubit(0, 1))]
    )
    grid_targets = spec.valid_targets.add()
    grid_targets.name = '2_qubit_anywhere'
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    gs_proto = spec.valid_gate_sets.add()

    gs_proto.name = 'cz_requires_three_qubits'

    gate = gs_proto.valid_gates.add()
    gate.id = 'cz'
    gate.valid_targets.extend(['2_qubit_anywhere'])
    gate.number_of_qubits = 3

    cz_gateset = _just_cz()
    with cirq.testing.assert_deprecated('Use cirq_google.GridDevice', deadline='v0.16', count=2):
        dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[cz_gateset])

        with pytest.raises(ValueError):
            dev.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))


def test_constrained_permutations():
    spec = device_pb2.DeviceSpecification()
    for row in range(5):
        for col in range(2):
            spec.valid_qubits.extend([v2.qubit_to_proto_id(cirq.GridQubit(row, col))])

    grid_targets = spec.valid_targets.add()
    grid_targets.name = 'meas_on_first_line'
    grid_targets.target_ordering = device_pb2.TargetSet.SUBSET_PERMUTATION
    new_target = grid_targets.targets.add()
    new_target.ids.extend([v2.qubit_to_proto_id(cirq.GridQubit(i, 0)) for i in range(5)])

    gs_proto = spec.valid_gate_sets.add()
    gs_proto.name = 'meas_set'

    gate = gs_proto.valid_gates.add()
    gate.id = 'meas'
    gate.valid_targets.extend(['meas_on_first_line'])

    meas_gateset = _just_meas()
    with cirq.testing.assert_deprecated('Use cirq_google.GridDevice', deadline='v0.16', count=2):
        dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[meas_gateset])

        dev.validate_operation(cirq.measure(cirq.GridQubit(0, 0)))
        dev.validate_operation(cirq.measure(cirq.GridQubit(1, 0)))
        dev.validate_operation(cirq.measure(cirq.GridQubit(2, 0)))
        dev.validate_operation(cirq.measure(*[cirq.GridQubit(i, 0) for i in range(5)]))

        with pytest.raises(ValueError):
            dev.validate_operation(cirq.measure(cirq.GridQubit(1, 1)))
        with pytest.raises(ValueError):
            dev.validate_operation(cirq.measure(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)))


def test_mixing_types():
    """Mixing SUBSET_PERMUTATION with SYMMETRIC targets is confusing,
    and not yet supported"""
    spec = device_pb2.DeviceSpecification()

    grid_targets = spec.valid_targets.add()
    grid_targets.name = 'subset'
    grid_targets.target_ordering = device_pb2.TargetSet.SUBSET_PERMUTATION

    grid_targets = spec.valid_targets.add()
    grid_targets.name = 'sym'
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC

    gs_proto = spec.valid_gate_sets.add()
    gs_proto.name = 'set_with_mismatched_targets'

    gate = gs_proto.valid_gates.add()
    gate.id = 'meas'
    gate.valid_targets.extend(['subset', 'sym'])

    meas_gateset = _just_meas()
    with cirq.testing.assert_deprecated('Use cirq_google.GridDevice', deadline='v0.16', count=1):
        with pytest.raises(NotImplementedError):
            _ = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[meas_gateset])


def test_serializable_device_str_named_qubits():
    with cirq.testing.assert_deprecated('Use cirq_google.GridDevice', deadline='v0.16', count=1):
        device = cg.SerializableDevice(
            qubits=[cirq.NamedQubit('a'), cirq.NamedQubit('b')], gate_definitions={}
        )
        assert device.__class__.__name__ in str(device)


def test_serializable_device_gate_definitions_filter():
    """Ignore items in gate_definitions dictionary with invalid keys."""
    with cirq.testing.assert_deprecated('Use cirq_google.GridDevice', deadline='v0.16', count=1):
        device = cg.SerializableDevice(
            qubits=[cirq.NamedQubit('a'), cirq.NamedQubit('b')],
            gate_definitions={cirq.FSimGate: [], cirq.NoiseModel: []},
        )
        # Two gates for cirq.FSimGate and the cirq.GlobalPhaseGate default
        assert len(device.metadata.gateset.gates) == 2

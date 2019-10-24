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

import copy
import pytest

import cirq
import cirq.google as cg
import cirq.google.api.v2 as v2
import cirq.google.api.v2.device_pb2 as device_pb2

_JUST_CZ = cg.SerializableGateSet(
    gate_set_name='cz_gate_set',
    serializers=[
        cg.GateOpSerializer(gate_type=cirq.CZPowGate,
                            serialized_gate_id='cz',
                            args=[])
    ],
    deserializers=[
        cg.GateOpDeserializer(serialized_gate_id='cz',
                              gate_constructor=cirq.CZPowGate,
                              args=[])
    ],
)

_JUST_MEAS = cg.SerializableGateSet(
    gate_set_name='meas_gate_set',
    serializers=[
        cg.GateOpSerializer(gate_type=cirq.MeasurementGate,
                            serialized_gate_id='meas',
                            args=[])
    ],
    deserializers=[
        cg.GateOpDeserializer(serialized_gate_id='meas',
                              gate_constructor=cirq.MeasurementGate,
                              args=[])
    ],
)


def test_foxtail():
    valid_qubit1 = cirq.GridQubit(0, 0)
    valid_qubit2 = cirq.GridQubit(1, 0)
    valid_qubit3 = cirq.GridQubit(1, 1)
    invalid_qubit1 = cirq.GridQubit(2, 2)
    invalid_qubit2 = cirq.GridQubit(2, 3)

    foxtail = cg.SerializableDevice.from_proto(
        proto=cg.known_devices.FOXTAIL_PROTO, gate_set=cg.gate_sets.XMON)
    foxtail.validate_operation(cirq.X(valid_qubit1))
    foxtail.validate_operation(cirq.X(valid_qubit2))
    foxtail.validate_operation(cirq.X(valid_qubit3))
    foxtail.validate_operation(cirq.XPowGate(exponent=0.1)(valid_qubit1))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.X(invalid_qubit1))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.X(invalid_qubit2))
    foxtail.validate_operation(cirq.CZ(valid_qubit1, valid_qubit2))
    foxtail.validate_operation(cirq.CZ(valid_qubit2, valid_qubit1))
    # Non-local
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.CZ(valid_qubit1, valid_qubit3))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.CZ(invalid_qubit1, invalid_qubit2))

    # Unsupport op
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.H(invalid_qubit1))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.H(valid_qubit1))

    # Measurement (any combination)
    foxtail.validate_operation(cirq.measure(valid_qubit1))
    foxtail.validate_operation(cirq.measure(valid_qubit1, valid_qubit2))
    foxtail.validate_operation(
        cirq.measure(valid_qubit3, valid_qubit1, valid_qubit2))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.measure(invalid_qubit1))


def test_mismatched_proto_serializer():
    augmented_proto = copy.deepcopy(cg.known_devices.FOXTAIL_PROTO)
    # Remove measurement gate
    del augmented_proto.valid_gate_sets[0].valid_gates[3]

    # Should throw value error that measurement gate is serialized
    # but not supported by the hardware
    with pytest.raises(ValueError):
        _ = cg.SerializableDevice.from_proto(proto=augmented_proto,
                                             gate_set=cg.gate_sets.XMON)


def test_named_qubit():
    augmented_proto = copy.deepcopy(cg.known_devices.FOXTAIL_PROTO)
    augmented_proto.valid_qubits.extend(["scooby_doo"])
    foxtail = cg.SerializableDevice.from_proto(proto=augmented_proto,
                                               gate_set=cg.gate_sets.XMON)
    foxtail.validate_operation(cirq.X(cirq.NamedQubit("scooby_doo")))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.X(cirq.NamedQubit("scrappy_doo")))


def test_duration_of():
    valid_qubit1 = cirq.GridQubit(0, 0)

    foxtail = cg.SerializableDevice.from_proto(
        proto=cg.known_devices.FOXTAIL_PROTO, gate_set=cg.gate_sets.XMON)

    assert foxtail.duration_of(cirq.X(valid_qubit1)) == cirq.Duration(nanos=20)

    # Unsupported op
    with pytest.raises(ValueError):
        assert foxtail.duration_of(cirq.H(valid_qubit1))


def test_assymetric_gate():
    spec = device_pb2.DeviceSpecification()
    for row in range(5):
        for col in range(2):
            spec.valid_qubits.extend(
                [v2.qubit_to_proto_id(cirq.GridQubit(row, col))])
    grid_targets = spec.valid_targets.add()
    grid_targets.name = 'left_to_right'
    grid_targets.target_ordering = device_pb2.TargetSet.ASYMMETRIC
    for row in range(5):
        new_target = grid_targets.targets.add()
        new_target.ids.extend([
            v2.qubit_to_proto_id(cirq.GridQubit(row, 0)),
            v2.qubit_to_proto_id(cirq.GridQubit(row, 1))
        ])

    gs_proto = spec.valid_gate_sets.add()
    gs_proto.name = 'cz_left_to_right_only'

    gate = gs_proto.valid_gates.add()
    gate.id = 'cz'
    gate.valid_targets.extend(['left_to_right'])

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_set=_JUST_CZ)

    for row in range(5):
        dev.validate_operation(
            cirq.CZ(cirq.GridQubit(row, 0), cirq.GridQubit(row, 1)))
        with pytest.raises(ValueError):
            dev.validate_operation(
                cirq.CZ(cirq.GridQubit(row, 1), cirq.GridQubit(row, 0)))


def test_unconstrained_gate():
    spec = device_pb2.DeviceSpecification()
    for row in range(5):
        for col in range(5):
            spec.valid_qubits.extend(
                [v2.qubit_to_proto_id(cirq.GridQubit(row, col))])
    grid_targets = spec.valid_targets.add()
    grid_targets.name = '2_qubit_anywhere'
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    gs_proto = spec.valid_gate_sets.add()
    gs_proto.name = 'cz_free_for_all'

    gate = gs_proto.valid_gates.add()
    gate.id = 'cz'
    gate.valid_targets.extend(['2_qubit_anywhere'])

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_set=_JUST_CZ)

    valid_qubit1 = cirq.GridQubit(4, 4)
    for row in range(4):
        for col in range(4):
            valid_qubit2 = cirq.GridQubit(row, col)
            dev.validate_operation(cirq.CZ(valid_qubit1, valid_qubit2))


def test_number_of_qubits_cz():
    spec = device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([
        v2.qubit_to_proto_id(cirq.GridQubit(0, 0)),
        v2.qubit_to_proto_id(cirq.GridQubit(0, 1))
    ])
    grid_targets = spec.valid_targets.add()
    grid_targets.name = '2_qubit_anywhere'
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    gs_proto = spec.valid_gate_sets.add()

    gs_proto.name = 'cz_requires_three_qubits'

    gate = gs_proto.valid_gates.add()
    gate.id = 'cz'
    gate.valid_targets.extend(['2_qubit_anywhere'])
    gate.number_of_qubits = 3

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_set=_JUST_CZ)

    with pytest.raises(ValueError):
        dev.validate_operation(
            cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))


def test_constrained_permutations():
    spec = device_pb2.DeviceSpecification()
    for row in range(5):
        for col in range(2):
            spec.valid_qubits.extend(
                [v2.qubit_to_proto_id(cirq.GridQubit(row, col))])

    grid_targets = spec.valid_targets.add()
    grid_targets.name = 'meas_on_first_line'
    grid_targets.target_ordering = device_pb2.TargetSet.SUBSET_PERMUTATION
    new_target = grid_targets.targets.add()
    new_target.ids.extend(
        [v2.qubit_to_proto_id(cirq.GridQubit(i, 0)) for i in range(5)])

    gs_proto = spec.valid_gate_sets.add()
    gs_proto.name = 'meas_set'

    gate = gs_proto.valid_gates.add()
    gate.id = 'meas'
    gate.valid_targets.extend(['meas_on_first_line'])

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_set=_JUST_MEAS)

    dev.validate_operation(cirq.measure(cirq.GridQubit(0, 0)))
    dev.validate_operation(cirq.measure(cirq.GridQubit(1, 0)))
    dev.validate_operation(cirq.measure(cirq.GridQubit(2, 0)))
    dev.validate_operation(
        cirq.measure(*[cirq.GridQubit(i, 0) for i in range(5)]))

    with pytest.raises(ValueError):
        dev.validate_operation(cirq.measure(cirq.GridQubit(1, 1)))
    with pytest.raises(ValueError):
        dev.validate_operation(
            cirq.measure(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)))


def test_mixing_types():
    """ Mixing SUBSET_PERMUTATION with SYMMETRIC targets is confusing,
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

    with pytest.raises(NotImplementedError):
        _ = cg.SerializableDevice.from_proto(proto=spec, gate_set=_JUST_MEAS)

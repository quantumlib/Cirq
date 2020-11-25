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
import textwrap
import unittest.mock as mock

import pytest

import cirq
import cirq.google as cg
import cirq.google.api.v2 as v2
import cirq.google.api.v2.device_pb2 as device_pb2
import cirq.google.common_serializers as cgc
import cirq.google.devices.known_devices as cgdk

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


def test_str_with_grid_qubits():
    qubits = cirq.GridQubit.rect(2, 3, left=1, top=1)
    device_proto = cgdk.create_device_proto_for_qubits(
        qubits=qubits,
        pairs=[
            (qubits[0], qubits[1]),
            (qubits[0], qubits[3]),
            (qubits[1], qubits[4]),
            (qubits[4], qubits[5]),
        ],
        gate_sets=[cg.FSIM_GATESET])
    device = cgdk.SerializableDevice.from_proto(device_proto,
                                                gate_sets=[cg.FSIM_GATESET])
    assert str(device) == textwrap.dedent("""\
        (1, 1)───(1, 2)   (1, 3)
        │        │
        │        │
        (2, 1)   (2, 2)───(2, 3)""")


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


def test_foxtail():
    valid_qubit1 = cirq.GridQubit(0, 0)
    valid_qubit2 = cirq.GridQubit(1, 0)
    valid_qubit3 = cirq.GridQubit(1, 1)
    invalid_qubit1 = cirq.GridQubit(2, 2)
    invalid_qubit2 = cirq.GridQubit(2, 3)

    foxtail = cg.SerializableDevice.from_proto(
        proto=cg.devices.known_devices.FOXTAIL_PROTO,
        gate_sets=[cg.gate_sets.XMON])
    assert foxtail.qubit_set() == frozenset(cirq.GridQubit.rect(2, 11, 0, 0))
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
    augmented_proto = copy.deepcopy(cg.devices.known_devices.FOXTAIL_PROTO)
    # Remove measurement gate
    del augmented_proto.valid_gate_sets[0].valid_gates[3]

    # Should throw value error that measurement gate is serialized
    # but not supported by the hardware
    with pytest.raises(ValueError):
        _ = cg.SerializableDevice.from_proto(proto=augmented_proto,
                                             gate_sets=[cg.gate_sets.XMON])


def test_named_qubit():
    augmented_proto = copy.deepcopy(cg.devices.known_devices.FOXTAIL_PROTO)
    augmented_proto.valid_qubits.extend(["scooby_doo"])
    foxtail = cg.SerializableDevice.from_proto(proto=augmented_proto,
                                               gate_sets=[cg.gate_sets.XMON])
    foxtail.validate_operation(cirq.X(cirq.NamedQubit("scooby_doo")))
    with pytest.raises(ValueError):
        foxtail.validate_operation(cirq.X(cirq.NamedQubit("scrappy_doo")))


def test_duration_of():
    valid_qubit1 = cirq.GridQubit(0, 0)

    foxtail = cg.SerializableDevice.from_proto(
        proto=cg.devices.known_devices.FOXTAIL_PROTO,
        gate_sets=[cg.gate_sets.XMON])

    assert foxtail.duration_of(cirq.X(valid_qubit1)) == cirq.Duration(nanos=15)

    # Unsupported op
    with pytest.raises(ValueError):
        assert foxtail.duration_of(cirq.H(valid_qubit1))


def test_asymmetric_gate():
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

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[_JUST_CZ])

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

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[_JUST_CZ])

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

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[_JUST_CZ])

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

    dev = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[_JUST_MEAS])

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
        _ = cg.SerializableDevice.from_proto(proto=spec, gate_sets=[_JUST_MEAS])


def test_multiple_gatesets():
    halfPiGateSet = cirq.google.serializable_gate_set.SerializableGateSet(
        gate_set_name='half_pi_gateset',
        serializers=cgc.SINGLE_QUBIT_HALF_PI_SERIALIZERS,
        deserializers=cgc.SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
    )
    allAnglesGateSet = cirq.google.serializable_gate_set.SerializableGateSet(
        gate_set_name='all_angles_gateset',
        serializers=cgc.SINGLE_QUBIT_SERIALIZERS,
        deserializers=cgc.SINGLE_QUBIT_DESERIALIZERS,
    )
    durations_dict = {'xy_pi': 20_000, 'xy_half_pi': 20_000, 'xy': 20_000}
    spec = cirq.google.devices.known_devices.create_device_proto_from_diagram(
        "aa\naa", [allAnglesGateSet, halfPiGateSet], durations_dict)
    dev = cg.SerializableDevice.from_proto(
        proto=spec, gate_sets=[allAnglesGateSet, halfPiGateSet])
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(1, 0)
    dev.validate_operation(cirq.X(q0))
    dev.validate_operation(cirq.X(q1))
    dev.validate_operation(cirq.XPowGate(exponent=0.1234)(q0))
    dev.validate_operation(cirq.XPowGate(exponent=0.2345)(q1))

    with pytest.raises(ValueError):
        dev.validate_operation(cirq.X(cirq.GridQubit(2, 2)))


def test_half_pi_takes_half_duration():
    """This tests that gate sets with two different definitions for the same
    gate perform correctly.  In this case, we set the XPowGate to be
    half the duration of the full exponent and make sure it still works.
    """
    half_pi_gs = cirq.google.serializable_gate_set.SerializableGateSet(
        gate_set_name='half_pi',
        serializers=[
            *cgc.SINGLE_QUBIT_HALF_PI_SERIALIZERS,
        ],
        deserializers=[
            *cgc.SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
        ],
    )
    durations_dict = {
        'xy_pi': 20_000,
        'xy_half_pi': 10_000,
    }
    spec = cirq.google.devices.known_devices.create_device_proto_from_diagram(
        "aa\naa", [half_pi_gs], durations_dict)

    # The gate set defines two different serializations for PhasedXPowGate
    device = cg.SerializableDevice.from_proto(proto=spec,
                                              gate_sets=[half_pi_gs])
    q = cirq.GridQubit(0, 0)
    pi = cirq.XPowGate(exponent=1.0)
    half_pi = cirq.XPowGate(exponent=0.5)
    assert device.duration_of(pi(q)) == cirq.Duration(picos=20_000)
    assert device.duration_of(half_pi(q)) == cirq.Duration(picos=10_000)


def test_multiple_fsim_gatesets():
    """This tests that gate sets with two different definitions for the same
    gate perform correctly.  In this case, we set the XPowGate to be
    half the duration of the full exponent and make sure it still works.
    """
    half_pi_gs = cirq.google.serializable_gate_set.SerializableGateSet(
        gate_set_name='half_pi',
        serializers=[
            *cgc.SINGLE_QUBIT_HALF_PI_SERIALIZERS,
        ],
        deserializers=[
            *cgc.SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
        ],
    )
    durations_dict = {
        'xy_pi': 20_000,
        'xy_half_pi': 10_000,
    }
    spec = cirq.google.devices.known_devices.create_device_proto_from_diagram(
        "aa\naa", [half_pi_gs], durations_dict)

    # The gate set defines two different serializations for PhasedXPowGate
    device = cg.SerializableDevice.from_proto(proto=spec,
                                              gate_sets=[half_pi_gs])
    q = cirq.GridQubit(0, 0)
    pi = cirq.XPowGate(exponent=1.0)
    half_pi = cirq.XPowGate(exponent=0.5)
    assert device.duration_of(pi(q)) == cirq.Duration(picos=20_000)
    assert device.duration_of(half_pi(q)) == cirq.Duration(picos=10_000)


def test_serializable_device_str_grid_qubits():
    spec = cirq.google.devices.known_devices.create_device_proto_from_diagram(
        "aa\naa", [cg.SYC_GATESET])
    device = cg.SerializableDevice.from_proto(proto=spec,
                                              gate_sets=[cg.SYC_GATESET])
    assert str(device) == """\
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)"""


def test_serializable_device_str_named_qubits():
    device = cg.SerializableDevice(
        qubits=[cirq.NamedQubit('a'),
                cirq.NamedQubit('b')],
        gate_definitions={})
    assert device.__class__.__name__ in str(device)


def test_sycamore23_str():
    assert str(cg.Sycamore23) == """\
                  (3, 2)
                  │
                  │
         (4, 1)───(4, 2)───(4, 3)
         │        │        │
         │        │        │
(5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)
         │        │        │        │
         │        │        │        │
         (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)
                  │        │        │        │
                  │        │        │        │
                  (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
                           │        │        │
                           │        │        │
                           (8, 3)───(8, 4)───(8, 5)
                                    │
                                    │
                                    (9, 4)"""

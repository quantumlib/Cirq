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
import numpy as np
import pytest

import cirq
import cirq_google
import cirq_google.common_serializers as cgc
import cirq_google.devices.known_devices as known_devices


def test_foxtail_qubits():
    expected_qubits = []
    for i in range(0, 2):
        for j in range(0, 11):
            expected_qubits.append(cirq.GridQubit(i, j))
    assert set(expected_qubits) == cirq_google.Foxtail.qubits


def test_foxtail_device_proto():
    assert (
        str(known_devices.FOXTAIL_PROTO)
        == """\
valid_gate_sets {
  name: "xmon"
  valid_gates {
    id: "xy"
    number_of_qubits: 1
    valid_args {
      name: "axis_half_turns"
      type: FLOAT
    }
    valid_args {
      name: "half_turns"
      type: FLOAT
    }
    gate_duration_picos: 15000
  }
  valid_gates {
    id: "z"
    number_of_qubits: 1
    valid_args {
      name: "half_turns"
      type: FLOAT
    }
    valid_args {
      name: "type"
      type: STRING
    }
  }
  valid_gates {
    id: "xyz"
    number_of_qubits: 1
    valid_args {
      name: "x_exponent"
      type: FLOAT
    }
    valid_args {
      name: "z_exponent"
      type: FLOAT
    }
    valid_args {
      name: "axis_phase_exponent"
      type: FLOAT
    }
  }
  valid_gates {
    id: "cz"
    number_of_qubits: 2
    valid_args {
      name: "half_turns"
      type: FLOAT
    }
    valid_args {
      name: "phase_match"
      type: STRING
    }
    gate_duration_picos: 45000
    valid_targets: "2_qubit_targets"
  }
  valid_gates {
    id: "meas"
    valid_args {
      name: "key"
      type: STRING
    }
    valid_args {
      name: "invert_mask"
      type: REPEATED_BOOLEAN
    }
    gate_duration_picos: 4000000
    valid_targets: "meas_targets"
  }
}
valid_qubits: "0_0"
valid_qubits: "0_1"
valid_qubits: "0_2"
valid_qubits: "0_3"
valid_qubits: "0_4"
valid_qubits: "0_5"
valid_qubits: "0_6"
valid_qubits: "0_7"
valid_qubits: "0_8"
valid_qubits: "0_9"
valid_qubits: "0_10"
valid_qubits: "1_0"
valid_qubits: "1_1"
valid_qubits: "1_2"
valid_qubits: "1_3"
valid_qubits: "1_4"
valid_qubits: "1_5"
valid_qubits: "1_6"
valid_qubits: "1_7"
valid_qubits: "1_8"
valid_qubits: "1_9"
valid_qubits: "1_10"
valid_targets {
  name: "meas_targets"
  target_ordering: SUBSET_PERMUTATION
}
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "0_1"
  }
  targets {
    ids: "0_0"
    ids: "1_0"
  }
  targets {
    ids: "0_1"
    ids: "0_2"
  }
  targets {
    ids: "0_1"
    ids: "1_1"
  }
  targets {
    ids: "0_2"
    ids: "0_3"
  }
  targets {
    ids: "0_2"
    ids: "1_2"
  }
  targets {
    ids: "0_3"
    ids: "0_4"
  }
  targets {
    ids: "0_3"
    ids: "1_3"
  }
  targets {
    ids: "0_4"
    ids: "0_5"
  }
  targets {
    ids: "0_4"
    ids: "1_4"
  }
  targets {
    ids: "0_5"
    ids: "0_6"
  }
  targets {
    ids: "0_5"
    ids: "1_5"
  }
  targets {
    ids: "0_6"
    ids: "0_7"
  }
  targets {
    ids: "0_6"
    ids: "1_6"
  }
  targets {
    ids: "0_7"
    ids: "0_8"
  }
  targets {
    ids: "0_7"
    ids: "1_7"
  }
  targets {
    ids: "0_8"
    ids: "0_9"
  }
  targets {
    ids: "0_8"
    ids: "1_8"
  }
  targets {
    ids: "0_9"
    ids: "0_10"
  }
  targets {
    ids: "0_9"
    ids: "1_9"
  }
  targets {
    ids: "0_10"
    ids: "1_10"
  }
  targets {
    ids: "1_0"
    ids: "1_1"
  }
  targets {
    ids: "1_1"
    ids: "1_2"
  }
  targets {
    ids: "1_2"
    ids: "1_3"
  }
  targets {
    ids: "1_3"
    ids: "1_4"
  }
  targets {
    ids: "1_4"
    ids: "1_5"
  }
  targets {
    ids: "1_5"
    ids: "1_6"
  }
  targets {
    ids: "1_6"
    ids: "1_7"
  }
  targets {
    ids: "1_7"
    ids: "1_8"
  }
  targets {
    ids: "1_8"
    ids: "1_9"
  }
  targets {
    ids: "1_9"
    ids: "1_10"
  }
}
"""
    )


def test_create_device_proto_for_irregular_grid():
    qubits = cirq.GridQubit.rect(2, 2)
    pairs = [
        (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
        (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
        (cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
    ]
    proto = known_devices.create_device_proto_for_qubits(qubits, pairs)
    assert (
        str(proto)
        == """\
valid_qubits: "0_0"
valid_qubits: "0_1"
valid_qubits: "1_0"
valid_qubits: "1_1"
valid_targets {
  name: "meas_targets"
  target_ordering: SUBSET_PERMUTATION
}
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "0_1"
  }
  targets {
    ids: "0_0"
    ids: "1_0"
  }
  targets {
    ids: "1_0"
    ids: "1_1"
  }
}
"""
    )


def test_multiple_gate_sets():
    halfPiGateSet = cirq_google.serializable_gate_set.SerializableGateSet(
        gate_set_name='half_pi_gateset',
        serializers=[*cgc.SINGLE_QUBIT_HALF_PI_SERIALIZERS, cgc.MEASUREMENT_SERIALIZER],
        deserializers=[*cgc.SINGLE_QUBIT_HALF_PI_DESERIALIZERS, cgc.MEASUREMENT_DESERIALIZER],
    )
    durations_dict = {
        'xy_pi': 20_000,
        'xy_half_pi': 10_000,
        'xy': 53_000,
        'cz': 11_000,
        'meas': 14_141,
    }
    test_proto = known_devices.create_device_proto_from_diagram(
        "aa\naa", [cirq_google.gate_sets.XMON, halfPiGateSet], durations_dict
    )
    assert (
        str(test_proto)
        == """\
valid_gate_sets {
  name: "xmon"
  valid_gates {
    id: "xy"
    number_of_qubits: 1
    valid_args {
      name: "axis_half_turns"
      type: FLOAT
    }
    valid_args {
      name: "half_turns"
      type: FLOAT
    }
    gate_duration_picos: 53000
  }
  valid_gates {
    id: "z"
    number_of_qubits: 1
    valid_args {
      name: "half_turns"
      type: FLOAT
    }
    valid_args {
      name: "type"
      type: STRING
    }
  }
  valid_gates {
    id: "xyz"
    number_of_qubits: 1
    valid_args {
      name: "x_exponent"
      type: FLOAT
    }
    valid_args {
      name: "z_exponent"
      type: FLOAT
    }
    valid_args {
      name: "axis_phase_exponent"
      type: FLOAT
    }
  }
  valid_gates {
    id: "cz"
    number_of_qubits: 2
    valid_args {
      name: "half_turns"
      type: FLOAT
    }
    valid_args {
      name: "phase_match"
      type: STRING
    }
    gate_duration_picos: 11000
    valid_targets: "2_qubit_targets"
  }
  valid_gates {
    id: "meas"
    valid_args {
      name: "key"
      type: STRING
    }
    valid_args {
      name: "invert_mask"
      type: REPEATED_BOOLEAN
    }
    gate_duration_picos: 14141
    valid_targets: "meas_targets"
  }
}
valid_gate_sets {
  name: "half_pi_gateset"
  valid_gates {
    id: "xy_pi"
    number_of_qubits: 1
    valid_args {
      name: "axis_half_turns"
      type: FLOAT
    }
    gate_duration_picos: 20000
  }
  valid_gates {
    id: "xy_half_pi"
    number_of_qubits: 1
    valid_args {
      name: "axis_half_turns"
      type: FLOAT
    }
    gate_duration_picos: 10000
  }
  valid_gates {
    id: "meas"
    valid_args {
      name: "key"
      type: STRING
    }
    valid_args {
      name: "invert_mask"
      type: REPEATED_BOOLEAN
    }
    gate_duration_picos: 14141
    valid_targets: "meas_targets"
  }
}
valid_qubits: "0_0"
valid_qubits: "0_1"
valid_qubits: "1_0"
valid_qubits: "1_1"
valid_targets {
  name: "meas_targets"
  target_ordering: SUBSET_PERMUTATION
}
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "0_1"
  }
  targets {
    ids: "0_0"
    ids: "1_0"
  }
  targets {
    ids: "0_1"
    ids: "1_1"
  }
  targets {
    ids: "1_0"
    ids: "1_1"
  }
}
"""
    )


def test_json_dict():
    assert cirq_google.Foxtail._json_dict_() == {
        'cirq_type': '_NamedConstantXmonDevice',
        'constant': 'cirq_google.Foxtail',
    }

    assert cirq_google.Bristlecone._json_dict_() == {
        'cirq_type': '_NamedConstantXmonDevice',
        'constant': 'cirq_google.Bristlecone',
    }

    with pytest.raises(ValueError, match='xmon device name'):
        known_devices._NamedConstantXmonDevice._from_json_dict_('the_unknown_fiddler')


@pytest.mark.parametrize('device', [cirq_google.Sycamore, cirq_google.Sycamore23])
def test_sycamore_devices(device):
    q0 = cirq.GridQubit(5, 3)
    q1 = cirq.GridQubit(5, 4)
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)
    device.validate_operation(syc)
    device.validate_operation(sqrt_iswap)
    assert device.duration_of(syc) == cirq.Duration(nanos=12)
    assert device.duration_of(sqrt_iswap) == cirq.Duration(nanos=32)


def test_sycamore_circuitop_device():
    circuitop_gateset = cirq.google.serializable_gate_set.SerializableGateSet(
        gate_set_name='circuitop_gateset',
        serializers=[cgc.CIRCUIT_OP_SERIALIZER],
        deserializers=[cgc.CIRCUIT_OP_DESERIALIZER],
    )
    gateset_list = [
        cirq.google.gate_sets.SQRT_ISWAP_GATESET,
        cirq.google.gate_sets.SYC_GATESET,
        circuitop_gateset,
    ]
    circuitop_proto = cirq.google.devices.known_devices.create_device_proto_from_diagram(
        known_devices._SYCAMORE23_GRID,
        gateset_list,
        known_devices._SYCAMORE_DURATIONS_PICOS,
    )
    device = cirq.google.SerializableDevice.from_proto(
        proto=circuitop_proto,
        gate_sets=gateset_list,
    )
    q0 = cirq.GridQubit(5, 3)
    q1 = cirq.GridQubit(5, 4)
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)
    circuit_op = cirq.CircuitOperation(cirq.FrozenCircuit(syc, sqrt_iswap))
    device.validate_operation(syc)
    device.validate_operation(sqrt_iswap)
    device.validate_operation(circuit_op)
    assert device.duration_of(syc) == cirq.Duration(nanos=12)
    assert device.duration_of(sqrt_iswap) == cirq.Duration(nanos=32)
    # CircuitOperations don't have a set duration.
    assert device.duration_of(circuit_op) == cirq.Duration(nanos=0)


def test_sycamore_grid_layout():
    # Qubits on Sycamore but not on Sycamore23
    q0 = cirq.GridQubit(5, 5)
    q1 = cirq.GridQubit(5, 6)
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)
    cirq_google.Sycamore.validate_operation(syc)
    cirq_google.Sycamore.validate_operation(sqrt_iswap)

    with pytest.raises(ValueError):
        cirq_google.Sycamore23.validate_operation(syc)
    with pytest.raises(ValueError):
        cirq_google.Sycamore23.validate_operation(sqrt_iswap)


def test_proto_with_circuitop():
    circuitop_gateset = cirq.google.serializable_gate_set.SerializableGateSet(
        gate_set_name='circuitop_gateset',
        serializers=[cgc.CIRCUIT_OP_SERIALIZER],
        deserializers=[cgc.CIRCUIT_OP_DESERIALIZER],
    )
    circuitop_proto = cirq.google.devices.known_devices.create_device_proto_from_diagram(
        "aa\naa",
        [circuitop_gateset],
    )

    assert (
        str(circuitop_proto)
        == """\
valid_gate_sets {
  name: "circuitop_gateset"
  valid_gates {
    id: "circuit"
  }
}
valid_qubits: "0_0"
valid_qubits: "0_1"
valid_qubits: "1_0"
valid_qubits: "1_1"
valid_targets {
  name: "meas_targets"
  target_ordering: SUBSET_PERMUTATION
}
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "0_1"
  }
  targets {
    ids: "0_0"
    ids: "1_0"
  }
  targets {
    ids: "0_1"
    ids: "1_1"
  }
  targets {
    ids: "1_0"
    ids: "1_1"
  }
}
"""
    )


def test_proto_with_waitgate():
    wait_gateset = cirq_google.serializable_gate_set.SerializableGateSet(
        gate_set_name='wait_gateset',
        serializers=[cgc.WAIT_GATE_SERIALIZER],
        deserializers=[cgc.WAIT_GATE_DESERIALIZER],
    )
    wait_proto = cirq_google.devices.known_devices.create_device_proto_from_diagram(
        "aa\naa",
        [wait_gateset],
    )
    wait_device = cirq_google.SerializableDevice.from_proto(
        proto=wait_proto, gate_sets=[wait_gateset]
    )
    q0 = cirq.GridQubit(1, 1)
    wait_op = cirq.wait(q0, nanos=25)
    wait_device.validate_operation(wait_op)

    assert (
        str(wait_proto)
        == """\
valid_gate_sets {
  name: "wait_gateset"
  valid_gates {
    id: "wait"
    number_of_qubits: 1
    valid_args {
      name: "nanos"
      type: FLOAT
    }
  }
}
valid_qubits: "0_0"
valid_qubits: "0_1"
valid_qubits: "1_0"
valid_qubits: "1_1"
valid_targets {
  name: "meas_targets"
  target_ordering: SUBSET_PERMUTATION
}
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "0_1"
  }
  targets {
    ids: "0_0"
    ids: "1_0"
  }
  targets {
    ids: "0_1"
    ids: "1_1"
  }
  targets {
    ids: "1_0"
    ids: "1_1"
  }
}
"""
    )


def test_adding_gates_multiple_times():
    waiting_for_godot = cirq_google.serializable_gate_set.SerializableGateSet(
        gate_set_name='wait_gateset',
        serializers=[cgc.WAIT_GATE_SERIALIZER, cgc.WAIT_GATE_SERIALIZER, cgc.WAIT_GATE_SERIALIZER],
        deserializers=[
            cgc.WAIT_GATE_DESERIALIZER,
            cgc.WAIT_GATE_DESERIALIZER,
            cgc.WAIT_GATE_DESERIALIZER,
        ],
    )
    wait_proto = cirq_google.devices.known_devices.create_device_proto_from_diagram(
        "aa",
        [waiting_for_godot],
    )
    wait_device = cirq_google.SerializableDevice.from_proto(
        proto=wait_proto, gate_sets=[waiting_for_godot]
    )
    q0 = cirq.GridQubit(0, 0)
    wait_op = cirq.wait(q0, nanos=25)
    wait_device.validate_operation(wait_op)

    assert (
        str(wait_proto)
        == """\
valid_gate_sets {
  name: "wait_gateset"
  valid_gates {
    id: "wait"
    number_of_qubits: 1
    valid_args {
      name: "nanos"
      type: FLOAT
    }
  }
}
valid_qubits: "0_0"
valid_qubits: "0_1"
valid_targets {
  name: "meas_targets"
  target_ordering: SUBSET_PERMUTATION
}
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "0_1"
  }
}
"""
    )

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
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
import cirq_google.devices.known_devices as known_devices
import cirq_google.serialization.common_serializers as cgc


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
    with cirq.testing.assert_deprecated('no longer be available', deadline='v0.16', count=1):
        halfPiGateSet = cirq_google.SerializableGateSet(
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
            "aa\naa", [cirq_google.XMON, halfPiGateSet], durations_dict
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
  valid_gates {
    id: "circuit"
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


def test_sycamore_circuitop_device():
    with cirq.testing.assert_deprecated('no longer be available', deadline='v0.16', count=1):
        circuitop_gateset = cirq_google.SerializableGateSet(
            gate_set_name='circuitop_gateset',
            serializers=[cgc.CIRCUIT_OP_SERIALIZER],
            deserializers=[cgc.CIRCUIT_OP_DESERIALIZER],
        )
        gateset_list = [cirq_google.SQRT_ISWAP_GATESET, cirq_google.SYC_GATESET, circuitop_gateset]
        circuitop_proto = cirq_google.devices.known_devices.create_device_proto_from_diagram(
            known_devices._SYCAMORE23_GRID, gateset_list, known_devices._SYCAMORE_DURATIONS_PICOS
        )
        device = cirq_google.SerializableDevice.from_proto(
            proto=circuitop_proto, gate_sets=gateset_list
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
    with cirq.testing.assert_deprecated('no longer be available', deadline='v0.16', count=1):
        circuitop_gateset = cirq_google.SerializableGateSet(
            gate_set_name='circuitop_gateset',
            serializers=[cgc.CIRCUIT_OP_SERIALIZER],
            deserializers=[cgc.CIRCUIT_OP_DESERIALIZER],
        )
        circuitop_proto = cirq_google.devices.known_devices.create_device_proto_from_diagram(
            "aa\naa", [circuitop_gateset]
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
    with cirq.testing.assert_deprecated('no longer be available', deadline='v0.16', count=1):
        wait_gateset = cirq_google.SerializableGateSet(
            gate_set_name='wait_gateset',
            serializers=[cgc.WAIT_GATE_SERIALIZER],
            deserializers=[cgc.WAIT_GATE_DESERIALIZER],
        )
        wait_proto = cirq_google.devices.known_devices.create_device_proto_from_diagram(
            "aa\naa", [wait_gateset]
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
    with cirq.testing.assert_deprecated('no longer be available', deadline='v0.16', count=1):
        waiting_for_godot = cirq_google.SerializableGateSet(
            gate_set_name='wait_gateset',
            serializers=[
                cgc.WAIT_GATE_SERIALIZER,
                cgc.WAIT_GATE_SERIALIZER,
                cgc.WAIT_GATE_SERIALIZER,
            ],
            deserializers=[
                cgc.WAIT_GATE_DESERIALIZER,
                cgc.WAIT_GATE_DESERIALIZER,
                cgc.WAIT_GATE_DESERIALIZER,
            ],
        )
        wait_proto = cirq_google.devices.known_devices.create_device_proto_from_diagram(
            "aa", [waiting_for_godot]
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


@pytest.mark.parametrize(
    'device, qubit_size, layout_str',
    [
        (
            cirq_google.Sycamore,
            88,
            """\
                                             (0, 5)───(0, 6)
                                             │        │
                                             │        │
                                    (1, 4)───(1, 5)───(1, 6)───(1, 7)
                                    │        │        │        │
                                    │        │        │        │
                           (2, 3)───(2, 4)───(2, 5)───(2, 6)───(2, 7)───(2, 8)
                           │        │        │        │        │        │
                           │        │        │        │        │        │
                  (3, 2)───(3, 3)───(3, 4)───(3, 5)───(3, 6)───(3, 7)───(3, 8)───(3, 9)
                  │        │        │        │        │        │        │        │
                  │        │        │        │        │        │        │        │
         (4, 1)───(4, 2)───(4, 3)───(4, 4)───(4, 5)───(4, 6)───(4, 7)───(4, 8)───(4, 9)
         │        │        │        │        │        │        │        │
         │        │        │        │        │        │        │        │
(5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)───(5, 5)───(5, 6)───(5, 7)───(5, 8)
         │        │        │        │        │        │        │
         │        │        │        │        │        │        │
         (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)───(6, 6)───(6, 7)
                  │        │        │        │        │
                  │        │        │        │        │
                  (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
                           │        │        │
                           │        │        │
                           (8, 3)───(8, 4)───(8, 5)
                                    │
                                    │
                                    (9, 4)""",
        ),
        (
            cirq_google.Sycamore23,
            32,
            """\
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
                                    (9, 4)""",
        ),
    ],
)
def test_sycamore_devices(device, qubit_size, layout_str):
    q0 = cirq.GridQubit(5, 3)
    q1 = cirq.GridQubit(5, 4)
    valid_sycamore_gates_and_ops = [
        cirq_google.SYC,
        cirq.SQRT_ISWAP,
        cirq.SQRT_ISWAP_INV,
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.Z(q0).with_tags(cirq_google.PhysicalZTag()),
        coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0),
        cirq.measure(q0),
        cirq.WaitGate(cirq.Duration(millis=5)),
        # TODO(#5050) Uncomment after GlobalPhaseGate support is added.
        # cirq.GlobalPhaseGate(-1.0),
    ]
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)

    assert str(device) == layout_str
    assert len(device.metadata.qubit_pairs) == qubit_size
    assert all(gate_or_op in device.metadata.gateset for gate_or_op in valid_sycamore_gates_and_ops)
    assert len(device.metadata.gate_durations) == len(device.metadata.gateset.gates)
    assert any(
        isinstance(cgs, cirq_google.SycamoreTargetGateset)
        for cgs in device.metadata.compilation_target_gatesets
    )
    assert any(
        isinstance(cgs, cirq.SqrtIswapTargetGateset)
        for cgs in device.metadata.compilation_target_gatesets
    )

    device.validate_operation(syc)
    device.validate_operation(sqrt_iswap)

    assert next(
        (
            duration
            for gate_family, duration in device.metadata.gate_durations.items()
            if syc in gate_family
        ),
        None,
    ) == cirq.Duration(nanos=12)
    assert next(
        (
            duration
            for gate_family, duration in device.metadata.gate_durations.items()
            if sqrt_iswap in gate_family
        ),
        None,
    ) == cirq.Duration(nanos=32)

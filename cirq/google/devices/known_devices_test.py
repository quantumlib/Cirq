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

import cirq
import cirq.google as cg
import cirq.google.common_serializers as cgc


def test_foxtail_qubits():
    expected_qubits = []
    for i in range(0, 2):
        for j in range(0, 11):
            expected_qubits.append(cirq.GridQubit(i, j))
    assert set(expected_qubits) == cirq.google.Foxtail.qubits


def test_foxtail_device_proto():
    assert str(cirq.google.devices.known_devices.FOXTAIL_PROTO) == """\
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
    gate_duration_picos: 20000
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
    id: "cz"
    number_of_qubits: 2
    valid_args {
      name: "half_turns"
      type: FLOAT
    }
    gate_duration_picos: 50000
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
    gate_duration_picos: 1000000
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


def test_multiple_gate_sets():
    halfPiGateSet = cg.serializable_gate_set.SerializableGateSet(
        gate_set_name='half_pi_gateset',
        serializers=[
            *cgc.SINGLE_QUBIT_HALF_PI_SERIALIZERS, cgc.MEASUREMENT_SERIALIZER
        ],
        deserializers=[
            *cgc.SINGLE_QUBIT_HALF_PI_DESERIALIZERS,
            cgc.MEASUREMENT_DESERIALIZER
        ],
    )
    durations_dict = {
        'xy_pi': 20_000,
        'xy_half_pi': 10_000,
        'xy': 53_000,
        'cz': 11_000,
        'meas': 14_141
    }
    test_proto = cg.devices.known_devices.create_device_proto_from_diagram(
        "aa\naa", [cg.gate_sets.XMON, halfPiGateSet], durations_dict)
    assert str(test_proto) == """\
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
    id: "cz"
    number_of_qubits: 2
    valid_args {
      name: "half_turns"
      type: FLOAT
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


def test_json_dict():
    assert cg.Foxtail._json_dict_() == {
        'cirq_type': '_NamedConstantXmonDevice',
        'constant': 'cirq.google.Foxtail',
        'measurement_duration': cirq.Duration(nanos=1000),
        'exp_w_duration': cirq.Duration(nanos=20),
        'exp_11_duration': cirq.Duration(nanos=50),
        'qubits': sorted(cirq.google.Foxtail.qubits)
    }

    assert cirq.google.Bristlecone._json_dict_() == {
        'cirq_type': '_NamedConstantXmonDevice',
        'constant': 'cirq.google.Bristlecone',
        'measurement_duration': cirq.Duration(nanos=1000),
        'exp_w_duration': cirq.Duration(nanos=20),
        'exp_11_duration': cirq.Duration(nanos=50),
        'qubits': sorted(cirq.google.Bristlecone.qubits)
    }


def test_sycamore_device():
    q0 = cirq.GridQubit(5, 4)
    q1 = cirq.GridQubit(5, 5)
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)
    cg.Sycamore.validate_operation(syc)
    cg.Sycamore.validate_operation(sqrt_iswap)
    assert cg.Sycamore.duration_of(syc) == cirq.Duration(nanos=12)
    assert cg.Sycamore.duration_of(sqrt_iswap) == cirq.Duration(nanos=32)

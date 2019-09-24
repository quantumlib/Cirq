# Copyright 2018 The Cirq Developers
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

from typing import Dict, List, Set, Tuple

from cirq.devices import GridQubit
from cirq.google import gate_sets, serializable_gate_set
from cirq.google.api.v2 import device_pb2
from cirq.google.xmon_device import XmonDevice
from cirq.ops import MeasurementGate, SingleQubitGate
from cirq.value import Duration


_2_QUBIT_TARGET_SET = "2_qubit_targets"
_MEAS_TARGET_SET = "meas_targets"


def _parse_device(s: str) -> Tuple[List[GridQubit], Dict[str, Set[GridQubit]]]:
    """Parse ASCIIart device layout into info about qubits and connectivity.

    Args:
        s: String representing the qubit layout. Each line represents a row,
            and each character in the row is a qubit, or a blank site if the
            character is a hyphen '-'. Different letters for the qubit specify
            which measurement line that qubit is connected to, e.g. all 'A'
            qubits share a measurement line. Leading and trailing spaces on
            each line are ignored.

    Returns:
        A list of qubits and a dict mapping measurement line name to the qubits
        on that measurement line.
    """
    lines = s.strip().split('\n')
    qubits = []  # type: List[GridQubit]
    measurement_lines = {}  # type: Dict[str, Set[GridQubit]]
    for row, line in enumerate(lines):
        for col, c in enumerate(line.strip()):
            if c != '-':
                qubit = GridQubit(row, col)
                qubits.append(qubit)
                measurement_line = measurement_lines.setdefault(c, set())
                measurement_line.add(qubit)
    return qubits, measurement_lines


def create_device_proto_from_diagram(
        s: str,
        gate_set: serializable_gate_set.SerializableGateSet = None,
        durations: Dict[str, int] = None,
) -> device_pb2.DeviceSpecification:
    """
    Parse ASCIIart device layout into DeviceSpecification proto containing
    information about qubits and targets.  Note that this does not populate
    the gate set information of the proto.

    Also accepts a gate set object that will create the gate set proto.

    This function also assumes that only adjacent qubits can be targets.
    """
    qubits, _ = _parse_device(s)
    spec = device_pb2.DeviceSpecification()
    meas_targets = spec.valid_targets.add()
    meas_targets.name = _MEAS_TARGET_SET
    meas_targets.target_ordering = device_pb2.TargetSet.SUBSET_PERMUTATION

    grid_targets = spec.valid_targets.add()
    grid_targets.name = _2_QUBIT_TARGET_SET
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    qubit_set = frozenset(qubits)
    neighbor_set: Set[Tuple] = set()
    spec.valid_qubits.extend([q.proto_id() for q in qubits])
    for q in qubits:
        for neighbor in sorted(q.neighbors(qubit_set)):
            if (neighbor, q) not in neighbor_set:
                # Don't add pairs twice
                new_target = grid_targets.targets.add()
                new_target.ids.extend((q.proto_id(), neighbor.proto_id()))
                neighbor_set.add((q, neighbor))
    if gate_set is not None:
        gs_proto = spec.valid_gate_sets.add()
        gs_proto.name = gate_set.gate_set_name
        gate_ids: Set[str] = set()
        for gate_type in gate_set.serializers:
            for serializer in gate_set.serializers[gate_type]:
                gate_id = serializer.serialized_gate_id
                if gate_id in gate_ids:
                    # Only add each type once
                    continue

                gate_ids.add(gate_id)
                gate = gs_proto.valid_gates.add()
                gate.id = gate_id
                if gate_type == MeasurementGate:
                    gate.valid_targets.extend([_MEAS_TARGET_SET])
                elif not issubclass(gate_type, SingleQubitGate):
                    gate.valid_targets.extend([_2_QUBIT_TARGET_SET])
                if durations is not None and gate.id in durations:
                    gate.gate_duration_picos = durations[gate.id]
                # TODO: add valid args and number of qubits
    return spec


_FOXTAIL_GRID = """
AAAAABBBBBB
CCCCCCDDDDD
"""


class _NamedConstantXmonDevice(XmonDevice):
    def __init__(self, constant: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._repr = constant

    def __repr__(self):
        return self._repr

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__,
            'constant': self._repr,
            'measurement_duration': self._measurement_duration,
            'exp_w_duration': self._exp_w_duration,
            'exp_11_duration': self._exp_z_duration,
            'qubits': sorted(self.qubits)
        }


Foxtail = _NamedConstantXmonDevice(
    'cirq.google.Foxtail',
    measurement_duration=Duration(nanos=1000),
    exp_w_duration=Duration(nanos=20),
    exp_11_duration=Duration(nanos=50),
    qubits=_parse_device(_FOXTAIL_GRID)[0])


# Duration dict in picoseconds
_DURATIONS_FOR_XMON = {
    'exp_11': 50000,
    'exp_w': 20000,
    'exp_z': 0,
    'meas': 1000000,
}

FOXTAIL_PROTO = create_device_proto_from_diagram(_FOXTAIL_GRID, gate_sets.XMON,
                                                 _DURATIONS_FOR_XMON)

_BRISTLECONE_GRID = """
-----AB-----
----ABCD----
---ABCDEF---
--ABCDEFGH--
-ABCDEFGHIJ-
ABCDEFGHIJKL
-CDEFGHIJKL-
--EFGHIJKL--
---GHIJKL---
----IJKL----
-----KL-----
"""


Bristlecone = _NamedConstantXmonDevice(
    'cirq.google.Bristlecone',
    measurement_duration=Duration(nanos=1000),
    exp_w_duration=Duration(nanos=20),
    exp_11_duration=Duration(nanos=50),
    qubits=_parse_device(_BRISTLECONE_GRID)[0])

BRISTLECONE_PROTO = create_device_proto_from_diagram(_BRISTLECONE_GRID,
                                                     gate_sets.XMON,
                                                     _DURATIONS_FOR_XMON)

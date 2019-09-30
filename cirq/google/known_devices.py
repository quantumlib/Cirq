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
from cirq.google.api import v2
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
        ascii_grid: str,
        gate_set: serializable_gate_set.SerializableGateSet = None,
        durations_picos: Dict[str, int] = None,
) -> device_pb2.DeviceSpecification:
    """
    Parse ASCIIart device layout into DeviceSpecification proto containing
    information about qubits and targets.  Note that this does not populate
    the gate set information of the proto.  This function also assumes that
    only adjacent qubits can be targets.

    Args:
            ascii_grid: An ASCII version of the grid, see _parse_device for
                    details
            gate_set: A serializable gate_set object that is used to define
                    the translation between gate ids and cirq Gate objects
            durations_picos: A dictionary with keys of gate ids and values of
                    the duration of that gate in picoseconds

    """

    qubits, _ = _parse_device(ascii_grid)
    spec = device_pb2.DeviceSpecification()

    # Create valid qubit list
    qubit_set = frozenset(qubits)
    spec.valid_qubits.extend([v2.qubit_to_proto_id(q) for q in qubits])

    # Set up a target set for measurement (any qubit permutation)
    meas_targets = spec.valid_targets.add()
    meas_targets.name = _MEAS_TARGET_SET
    meas_targets.target_ordering = device_pb2.TargetSet.SUBSET_PERMUTATION

    # Set up a target set for 2 qubit gates (all adjacent pairs)
    grid_targets = spec.valid_targets.add()
    grid_targets.name = _2_QUBIT_TARGET_SET
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC

    # Create the target set as all adjacent pairs on the grid
    neighbor_set: Set[Tuple] = set()
    for q in qubits:
        for neighbor in sorted(q.neighbors(qubit_set)):
            if (neighbor, q) not in neighbor_set:
                # Don't add pairs twice
                new_target = grid_targets.targets.add()
                new_target.ids.extend(
                    (v2.qubit_to_proto_id(q), v2.qubit_to_proto_id(neighbor)))
                neighbor_set.add((q, neighbor))

    # Create gate set
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

                # Choose target set and number of qubits based on gate type.

                # Note: if it is not a measurement gate and doesn't inherit
                # from SingleQubitGate, it is assumed to be a two qubit gate.
                if gate_type == MeasurementGate:
                    gate.valid_targets.extend([_MEAS_TARGET_SET])
                elif issubclass(gate_type, SingleQubitGate):
                    gate.number_of_qubits = 1
                else:
                    # This must be a two-qubit gate
                    gate.valid_targets.extend([_2_QUBIT_TARGET_SET])
                    gate.number_of_qubits = 2

                # Add gate duration
                if durations_picos is not None and gate.id in durations_picos:
                    gate.gate_duration_picos = durations_picos[gate.id]

                # Add argument names and types for each gate.
                for arg in serializer.args:
                    new_arg = gate.valid_args.add()
                    if arg.serialized_type == str:
                        new_arg.type = device_pb2.ArgDefinition.STRING
                    if arg.serialized_type == float:
                        new_arg.type = device_pb2.ArgDefinition.FLOAT
                    if arg.serialized_type == List[bool]:
                        new_arg.type = device_pb2.ArgDefinition.REPEATED_BOOLEAN
                    new_arg.name = arg.serialized_name
                    # Note: this does not yet support adding allowed_ranges

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
    'exp_11': 50_000,
    'exp_w': 20_000,
    'exp_z': 0,
    'meas': 1_000_000,
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

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

from typing import Any, Collection, Dict, Optional, Iterable, List, Set, Tuple

import cirq
from cirq import _compat
from cirq._doc import document
from cirq_google.api import v2
from cirq_google.api.v2 import device_pb2
from cirq_google.devices.serializable_device import SerializableDevice
from cirq_google.devices.xmon_device import _XmonDeviceBase
from cirq_google.serialization import gate_sets, op_serializer, serializable_gate_set

_2_QUBIT_TARGET_SET = "2_qubit_targets"
_MEAS_TARGET_SET = "meas_targets"


def _parse_device(s: str) -> Tuple[List[cirq.GridQubit], Dict[str, Set[cirq.GridQubit]]]:
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
    qubits: List[cirq.GridQubit] = []
    measurement_lines: Dict[str, Set[cirq.GridQubit]] = {}
    for row, line in enumerate(lines):
        for col, c in enumerate(line.strip()):
            if c != '-':
                qubit = cirq.GridQubit(row, col)
                qubits.append(qubit)
                measurement_line = measurement_lines.setdefault(c, set())
                measurement_line.add(qubit)
    return qubits, measurement_lines


def create_device_proto_from_diagram(
    ascii_grid: str,
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
    durations_picos: Optional[Dict[str, int]] = None,
    out: Optional[device_pb2.DeviceSpecification] = None,
) -> device_pb2.DeviceSpecification:
    """Parse ASCIIart device layout into DeviceSpecification proto.

    This function assumes that all pairs of adjacent qubits are valid targets
    for two-qubit gates.

    Args:
        ascii_grid: ASCII version of the grid (see _parse_device for details).
        gate_sets: Gate sets that define the translation between gate ids and
            cirq Gate objects.
        durations_picos: A map from gate ids to gate durations in picoseconds.
        out: If given, populate this proto, otherwise create a new proto.
    """
    qubits, _ = _parse_device(ascii_grid)

    # Create a list of all adjacent pairs on the grid for two-qubit gates.
    qubit_set = frozenset(qubits)
    pairs: List[Tuple[cirq.Qid, cirq.Qid]] = []
    for qubit in qubits:
        for neighbor in sorted(qubit.neighbors()):
            if neighbor > qubit and neighbor in qubit_set:
                pairs.append((qubit, neighbor))

    return create_device_proto_for_qubits(qubits, pairs, gate_sets, durations_picos, out)


def create_device_proto_for_qubits(
    qubits: Collection[cirq.Qid],
    pairs: Collection[Tuple[cirq.Qid, cirq.Qid]],
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
    durations_picos: Optional[Dict[str, int]] = None,
    out: Optional[device_pb2.DeviceSpecification] = None,
) -> device_pb2.DeviceSpecification:
    """Create device spec for the given qubits and coupled pairs.

    Args:
        qubits: Qubits that can perform single-qubit gates.
        pairs: Pairs of coupled qubits that can perform two-qubit gates.
        gate_sets: Gate sets that define the translation between gate ids and
            cirq Gate objects.
        durations_picos: A map from gate ids to gate durations in picoseconds.
        out: If given, populate this proto, otherwise create a new proto.
    """
    if out is None:
        out = device_pb2.DeviceSpecification()

    # Create valid qubit list
    out.valid_qubits.extend(v2.qubit_to_proto_id(q) for q in qubits)

    # Set up a target set for measurement (any qubit permutation)
    meas_targets = out.valid_targets.add()
    meas_targets.name = _MEAS_TARGET_SET
    meas_targets.target_ordering = device_pb2.TargetSet.SUBSET_PERMUTATION

    # Set up a target set for 2 qubit gates (specified qubit pairs)
    grid_targets = out.valid_targets.add()
    grid_targets.name = _2_QUBIT_TARGET_SET
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    for pair in pairs:
        new_target = grid_targets.targets.add()
        new_target.ids.extend(v2.qubit_to_proto_id(q) for q in pair)

    # Create gate sets
    arg_def = device_pb2.ArgDefinition
    for gate_set in gate_sets or []:
        gs_proto = out.valid_gate_sets.add()
        gs_proto.name = gate_set.name
        gate_ids: Set[str] = set()
        for internal_type in gate_set.serializers:
            for serializer in gate_set.serializers[internal_type]:
                gate_id = serializer.serialized_id
                if gate_id in gate_ids:
                    # Only add each type once
                    continue

                gate_ids.add(gate_id)
                gate = gs_proto.valid_gates.add()
                gate.id = gate_id

                if not isinstance(serializer, op_serializer.GateOpSerializer):
                    # This implies that 'serializer' handles non-gate ops,
                    # such as CircuitOperations. No other properties apply.
                    continue

                # Choose target set and number of qubits based on gate type.
                gate_type = internal_type

                # Note: if it is not a measurement gate and doesn't inherit
                # from SingleQubitGate, it's assumed to be a two qubit gate.
                if gate_type == cirq.MeasurementGate:
                    gate.valid_targets.append(_MEAS_TARGET_SET)
                elif gate_type == cirq.WaitGate:
                    # TODO: Refactor gate-sets / device to eliminate the need
                    # to keep checking type here.
                    # Github issue:
                    # https://github.com/quantumlib/Cirq/issues/2537
                    gate.number_of_qubits = 1
                elif issubclass(gate_type, cirq.SingleQubitGate):
                    gate.number_of_qubits = 1
                else:
                    # This must be a two-qubit gate
                    gate.valid_targets.append(_2_QUBIT_TARGET_SET)
                    gate.number_of_qubits = 2

                # Add gate duration
                if durations_picos is not None and gate.id in durations_picos:
                    gate.gate_duration_picos = durations_picos[gate.id]

                # Add argument names and types for each gate.
                for arg in serializer.args:
                    new_arg = gate.valid_args.add()
                    if arg.serialized_type == str:
                        new_arg.type = arg_def.STRING
                    if arg.serialized_type == float:
                        new_arg.type = arg_def.FLOAT
                    if arg.serialized_type == List[bool]:
                        new_arg.type = arg_def.REPEATED_BOOLEAN
                    new_arg.name = arg.serialized_name
                    # Note: this does not yet support adding allowed_ranges

    return out


_FOXTAIL_GRID = """
AAAAABBBBBB
CCCCCCDDDDD
"""


class _NamedConstantXmonDevice(_XmonDeviceBase):
    def __init__(self, constant: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._repr = constant

    def __repr__(self) -> str:
        return self._repr

    @classmethod
    def _from_json_dict_(cls, constant: str, **kwargs):
        _compat._warn_or_error(
            f'NamedConstantXmonDevice was used but is deprecated.\n'
            f'It will be removed in cirq v0.15.\n'
        )
        if constant in ['cirq.google.Foxtail', 'cirq_google.Foxtail']:
            return Foxtail
        if constant in ['cirq.google.Bristlecone', 'cirq_google.Bristlecone']:
            return Bristlecone
        raise ValueError(f'Unrecognized xmon device name: {constant!r}')

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'constant': self._repr,
        }


Foxtail = _NamedConstantXmonDevice(
    'cirq_google.Foxtail',
    measurement_duration=cirq.Duration(nanos=4000),
    exp_w_duration=cirq.Duration(nanos=20),
    exp_11_duration=cirq.Duration(nanos=50),
    qubits=_parse_device(_FOXTAIL_GRID)[0],
)
document(
    Foxtail,
    f"""72 xmon qubit device.

**Qubit grid**:
```
{str(Foxtail)}
```
""",
)

# Duration dict in picoseconds
_DURATIONS_FOR_XMON = {
    'cz': 45_000,
    'xy': 15_000,
    'z': 0,
    'meas': 4_000_000,  # 1000ns for readout, 3000ns for "ring down"
}

FOXTAIL_PROTO = create_device_proto_from_diagram(
    _FOXTAIL_GRID, [gate_sets.XMON], _DURATIONS_FOR_XMON
)

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
    'cirq_google.Bristlecone',
    measurement_duration=cirq.Duration(nanos=4000),
    exp_w_duration=cirq.Duration(nanos=20),
    exp_11_duration=cirq.Duration(nanos=50),
    qubits=_parse_device(_BRISTLECONE_GRID)[0],
)

document(
    Bristlecone,
    f"""72 xmon qubit device.

**Qubit grid**:
```
{str(Bristlecone)}
```
""",
)

BRISTLECONE_PROTO = create_device_proto_from_diagram(
    _BRISTLECONE_GRID, [gate_sets.XMON], _DURATIONS_FOR_XMON
)

_SYCAMORE_GRID = """
-----AB---
----ABCD--
---ABCDEF-
--ABCDEFGH
-ABCDEFGHI
ABCDEFGHI-
-CDEFGHI--
--EFGHI---
---GHI----
----I-----
"""

_SYCAMORE_DURATIONS_PICOS = {
    'xy': 25_000,
    'xy_half_pi': 25_000,
    'xy_pi': 25_000,
    'xyz': 25_000,
    'fsim_pi_4': 32_000,
    'inv_fsim_pi_4': 32_000,
    'syc': 12_000,
    'z': 0,
    'meas': 4_000_000,  # 1000 ns for readout, 3000ns for ring_down
}

SYCAMORE_PROTO = create_device_proto_from_diagram(
    _SYCAMORE_GRID,
    [gate_sets.SQRT_ISWAP_GATESET, gate_sets.SYC_GATESET],
    _SYCAMORE_DURATIONS_PICOS,
)

Sycamore = SerializableDevice.from_proto(
    proto=SYCAMORE_PROTO, gate_sets=[gate_sets.SQRT_ISWAP_GATESET, gate_sets.SYC_GATESET]
)

# Subset of the Sycamore grid with a reduced layout.
_SYCAMORE23_GRID = """
----------
----------
----------
--A-------
-ABC------
ABCDE-----
-CDEFG----
--EFGHI---
---GHI----
----I-----
"""

SYCAMORE23_PROTO = create_device_proto_from_diagram(
    _SYCAMORE23_GRID,
    [gate_sets.SQRT_ISWAP_GATESET, gate_sets.SYC_GATESET],
    _SYCAMORE_DURATIONS_PICOS,
)

Sycamore23 = SerializableDevice.from_proto(
    proto=SYCAMORE23_PROTO, gate_sets=[gate_sets.SQRT_ISWAP_GATESET, gate_sets.SYC_GATESET]
)

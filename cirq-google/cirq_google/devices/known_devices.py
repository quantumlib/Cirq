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

from typing import Collection, Dict, Optional, List, Set, Tuple, cast

import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import device_pb2
from cirq_google.devices import grid_device
from cirq_google.experimental.ops import coupler_pulse
from cirq_google.ops import physical_z_tag, sycamore_gate

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


def _create_grid_device_from_diagram(
    ascii_grid: str,
    gateset: cirq.Gateset,
    gate_durations: Optional[Dict['cirq.GateFamily', 'cirq.Duration']] = None,
    out: Optional[device_pb2.DeviceSpecification] = None,
) -> grid_device.GridDevice:
    """Parse ASCIIart device layout into a GridDevice instance.

    This function assumes that all pairs of adjacent qubits are valid targets
    for two-qubit gates.

    Args:
        ascii_grid: ASCII version of the grid (see _parse_device for details).
        gateset: The device's gate set.
        gate_durations: A map of durations for each gate in the gate set.
        out: If given, populate this proto, otherwise create a new proto.
    """
    qubits, _ = _parse_device(ascii_grid)

    # Create a list of all adjacent pairs on the grid for two-qubit gates.
    qubit_set = frozenset(qubits)
    pairs: List[Tuple[cirq.GridQubit, cirq.GridQubit]] = []
    for qubit in qubits:
        for neighbor in sorted(qubit.neighbors()):
            if neighbor > qubit and neighbor in qubit_set:
                pairs.append((qubit, cast(cirq.GridQubit, neighbor)))

    device_specification = grid_device._create_device_specification_proto(
        qubits=qubits, pairs=pairs, gateset=gateset, gate_durations=gate_durations, out=out
    )
    return grid_device.GridDevice.from_proto(device_specification)


def populate_qubits_in_device_proto(
    qubits: Collection[cirq.Qid], out: device_pb2.DeviceSpecification
) -> None:
    """Populates `DeviceSpecification.valid_qubits` with the device's qubits.

    Args:
        qubits: The collection of the device's qubits.
        out: The `DeviceSpecification` to be populated.
    """
    out.valid_qubits.extend(v2.qubit_to_proto_id(q) for q in qubits)


def populate_qubit_pairs_in_device_proto(
    pairs: Collection[Tuple[cirq.Qid, cirq.Qid]], out: device_pb2.DeviceSpecification
) -> None:
    """Populates `DeviceSpecification.valid_targets` with the device's qubit pairs.

    Args:
        pairs: The collection of the device's bi-directional qubit pairs.
        out: The `DeviceSpecification` to be populated.
    """
    grid_targets = out.valid_targets.add()
    grid_targets.name = _2_QUBIT_TARGET_SET
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    for pair in pairs:
        new_target = grid_targets.targets.add()
        new_target.ids.extend(v2.qubit_to_proto_id(q) for q in pair)


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


# Deprecated: replaced by _SYCAMORE_DURATIONS
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


_SYCAMORE_GATESET = cirq.Gateset(
    sycamore_gate.SYC,
    cirq.SQRT_ISWAP,
    cirq.SQRT_ISWAP_INV,
    cirq.PhasedXZGate,
    # Physical Z and virtual Z gates are represented separately because they
    # have different gate durations.
    cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[physical_z_tag.PhysicalZTag()]),
    cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[physical_z_tag.PhysicalZTag()]),
    coupler_pulse.CouplerPulse,
    cirq.MeasurementGate,
    cirq.WaitGate,
)


_SYCAMORE_DURATIONS = {
    cirq.GateFamily(sycamore_gate.SYC): cirq.Duration(nanos=12),
    cirq.GateFamily(cirq.SQRT_ISWAP): cirq.Duration(nanos=32),
    cirq.GateFamily(cirq.SQRT_ISWAP_INV): cirq.Duration(nanos=32),
    cirq.GateFamily(cirq.ops.phased_x_z_gate.PhasedXZGate): cirq.Duration(nanos=25),
    cirq.GateFamily(
        cirq.ops.common_gates.ZPowGate, tags_to_ignore=[physical_z_tag.PhysicalZTag()]
    ): cirq.Duration(nanos=0),
    cirq.GateFamily(
        cirq.ops.common_gates.ZPowGate, tags_to_accept=[physical_z_tag.PhysicalZTag()]
    ): cirq.Duration(nanos=20),
    cirq.GateFamily(cirq.ops.measurement_gate.MeasurementGate): cirq.Duration(millis=4),
}


Sycamore = _create_grid_device_from_diagram(_SYCAMORE_GRID, _SYCAMORE_GATESET, _SYCAMORE_DURATIONS)


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


Sycamore23 = _create_grid_device_from_diagram(
    _SYCAMORE23_GRID, _SYCAMORE_GATESET, _SYCAMORE_DURATIONS
)

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

"""A combination of several optimizations targeting XmonDevice."""
from typing import Callable, cast, List, Optional, TYPE_CHECKING

from cirq import circuits, devices, optimizers
from cirq.google import convert_to_xmon_gates, xmon_device

if TYPE_CHECKING:
    import cirq

_TOLERANCE = 1e-5


def _merge_rots(c: 'cirq.Circuit'):
    return optimizers.merge_single_qubit_gates_into_phased_x_z(c, _TOLERANCE)


_OPTIMIZERS: List[Callable[['cirq.Circuit'], None]] = [
    convert_to_xmon_gates.ConvertToXmonGates().optimize_circuit,
    optimizers.MergeInteractions(tolerance=_TOLERANCE,
                                 allow_partial_czs=False).optimize_circuit,
    _merge_rots,
    optimizers.EjectPhasedPaulis(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.EjectZ(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.DropNegligible(tolerance=_TOLERANCE).optimize_circuit,
]

_OPTIMIZERS_PART_CZ: List[Callable[['cirq.Circuit'], None]] = [
    convert_to_xmon_gates.ConvertToXmonGates().optimize_circuit,
    optimizers.MergeInteractions(tolerance=_TOLERANCE,
                                 allow_partial_czs=True).optimize_circuit,
    _merge_rots,
    optimizers.EjectPhasedPaulis(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.EjectZ(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.DropNegligible(tolerance=_TOLERANCE).optimize_circuit,
]


def optimized_for_xmon(
        circuit: 'cirq.Circuit',
        new_device: Optional[xmon_device.XmonDevice] = None,
        qubit_map: Callable[['cirq.Qid'], devices.GridQubit] = lambda e: cast(
            devices.GridQubit, e),
        allow_partial_czs: bool = False,
) -> 'cirq.Circuit':
    """Optimizes a circuit with XmonDevice in mind.

    Starts by converting the circuit's operations to the xmon gate set, then
    begins merging interactions and rotations, ejecting pi-rotations and phasing
    operations, dropping unnecessary operations, and pushing operations earlier.

    Args:
        circuit: The circuit to optimize.
        new_device: The device the optimized circuit should be targeted at. If
            set to None, the circuit's current device is used.
        qubit_map: Transforms the qubits (e.g. so that they are GridQubits).
        allow_partial_czs: If true, the optimized circuit may contain partial CZ
            gates.  Otherwise all partial CZ gates will be converted to full CZ
            gates.  At worst, two CZ gates will be put in place of each partial
            CZ from the input.

    Returns:
        The optimized circuit.
    """
    copy = circuit.copy()
    opts = _OPTIMIZERS_PART_CZ if allow_partial_czs else _OPTIMIZERS
    for optimizer in opts:
        optimizer(copy)

    return circuits.Circuit(
        (op.transform_qubits(qubit_map) for op in copy.all_operations()),
        strategy=circuits.InsertStrategy.EARLIEST,
        device=new_device or copy.device)

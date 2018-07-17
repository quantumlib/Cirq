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

from cirq import circuits
from cirq.google import (
    convert_to_xmon_gates,
    merge_rotations,
    merge_interactions,
    eject_full_w,
    eject_z,
)

_TOLERANCE = 1e-5

_OPTIMIZERS = [
    convert_to_xmon_gates.ConvertToXmonGates(),

    merge_interactions.MergeInteractions(tolerance=_TOLERANCE),
    eject_full_w.EjectFullW(tolerance=_TOLERANCE),
    eject_z.EjectZ(tolerance=_TOLERANCE),
    merge_rotations.MergeRotations(tolerance=_TOLERANCE),
    circuits.DropNegligible(tolerance=_TOLERANCE),
]


def optimized_for_xmon(circuit: circuits.Circuit) -> circuits.Circuit:
    """Optimizes a circuit with XmonDevice in mind.

    Starts by converting the circuit's operations to the xmon gate set, then
    begins merging interactions and rotations, ejecting pi-rotations and phasing
    operations, dropping unnecessary operations, and pushing operations earlier.

    Args:
        circuit: The circuit to optimize.

    Returns:
        The optimized circuit.
    """
    copy = circuit.copy()
    for optimizer in _OPTIMIZERS:
        optimizer.optimize_circuit(copy)

    return circuits.Circuit().from_ops(
        copy.all_operations(),
        strategy=circuits.InsertStrategy.EARLIEST)

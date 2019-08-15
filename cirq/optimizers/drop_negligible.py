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

"""An optimization pass that removes operations with tiny effects."""

from typing import List, Tuple, TYPE_CHECKING

from cirq import protocols
from cirq.circuits import circuit as _circuit

if TYPE_CHECKING:
    from cirq import ops


class DropNegligible():
    """An optimization pass that removes operations with tiny effects."""

    def __init__(self, tolerance: float = 1e-8) -> None:
        self.tolerance = tolerance

    def __call__(self, circuit: _circuit.Circuit):
        self.optimize_circuit(circuit)

    def optimize_circuit(self, circuit: _circuit.Circuit) -> None:
        deletions: List[Tuple[int, ops.Operation]] = []
        for moment_index, moment in enumerate(circuit):
            for op in moment.operations:
                if (op is not None and
                        protocols.trace_distance_bound(op) <= self.tolerance):
                    deletions.append((moment_index, op))
        circuit.batch_remove(deletions)

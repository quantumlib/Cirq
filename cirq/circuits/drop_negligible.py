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
from typing import Optional

from cirq import ops, extension
from cirq.circuits import optimization_pass, circuit as _circuit


class DropNegligible(optimization_pass.PointOptimizer):
    """An optimization pass that removes operations with tiny effects."""

    def __init__(self,
                 tolerance: float = 1e-8,
                 extensions: extension.Extensions = None) -> None:
        self.tolerance = tolerance
        self.extensions = extensions or extension.Extensions()

    def optimization_at(
            self,
            circuit: _circuit.Circuit,
            index: int,
            op: ops.Operation
            ) -> Optional[optimization_pass.PointOptimizationSummary]:

        gate = self.extensions.try_cast(ops.BoundedEffect, op.gate)
        if gate is None or gate.trace_distance_bound() > self.tolerance:
            return None

        return optimization_pass.PointOptimizationSummary(
            clear_span=1,
            new_operations=(),
            clear_qubits=op.qubits)

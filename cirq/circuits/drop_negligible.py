# Copyright 2017 Google LLC
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

from cirq import ops
from cirq.circuits.optimization_pass import PointOptimizer


class DropNegligible(PointOptimizer):
    """An optimization pass that removes operations with tiny effects."""

    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance

    def optimize_at(self, circuit, index, op):
        if (isinstance(op.gate, ops.BoundedEffectGate) and
                op.gate.trace_distance_bound() <= self.tolerance):
            circuit.clear_operations_touching(op.qubits, [index])

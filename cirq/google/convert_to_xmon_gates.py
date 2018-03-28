# Copyright 2018 Google LLC
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

from cirq import ops
from cirq.circuits import InsertStrategy
from cirq.google.xmon_gate_extensions import xmon_gate_ext
from cirq.google.xmon_gates import XmonGate
from cirq.circuits.optimization_pass import PointOptimizer


class ConvertToXmonGates(PointOptimizer):
    """Pointwise converts a circuit to XmonGates if possible."""

    def __init__(self, ignore_failures=True):
        self.ignore_failures = ignore_failures

    def optimize_at(self, circuit, index, op):
        if self.ignore_failures:
            c = xmon_gate_ext.try_cast(op.gate, XmonGate)
            if c is None:
                return
        else:
            c = xmon_gate_ext.cast(op.gate, XmonGate)

        if c is not op.gate:
            circuit.clear_operations_touching(op.qubits, [index])
            return circuit.insert(index,
                                  ops.Operation(c, op.qubits),
                                  strategy=InsertStrategy.INLINE)

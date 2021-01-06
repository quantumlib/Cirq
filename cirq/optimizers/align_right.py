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

"""An optimization pass that aligns gates to the right of the circuit."""

from cirq import circuits, ops

class AlignRight:

    def optimize_circuit(self, circuit: circuits.Circuit):
      next = circuits.Circuit(ops.freeze_op_tree(circuit)[::-1])[::-1]
      deletions: List[Tuple[int, ops.Operation]] = []
      for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
          deletions.append((moment_index, op))
      circuit.batch_remove(deletions)
      
      for moment_index, moment in enumerate(next):
        for op in moment.operations:
          circuit.insert(moment_index + 1, op, strategy=circuits.InsertStrategy.INLINE)

# Copyright 2018 The ops Developers
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

from cirq import ops, circuits
from cirq.contrib.paulistring.pauli_string_optimize import (
    pauli_string_optimized_circuit)
from cirq.contrib.paulistring.clifford_optimize import (
    clifford_optimized_circuit)


def optimized_circuit(circuit: circuits.Circuit,
                      tolerance: float = 1e-8,
                      repeat: int = 10,
                      ) -> circuits.Circuit:
    for _ in range(repeat):
        circuit2 = pauli_string_optimized_circuit(
                        circuit,
                        move_cliffords=True,
                        tolerance=tolerance)
        circuit3 = clifford_optimized_circuit(
                        circuit2,
                        tolerance=tolerance)
        if (len(circuit3) == len(circuit)
            and _cz_count(circuit3) == _cz_count(circuit)):
            return circuit3
        circuit = circuit3
        print('REPEAT')
    return circuit


def _cz_count(circuit):
    return sum(isinstance(op, ops.GateOperation)
               and isinstance(op, ops.Rot11Gate)
               for op in circuit)

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

from typing import cast

from cirq import circuits, ops, protocols


def pauli_string_reorder_pred(op1: ops.Operation, op2: ops.Operation) -> bool:
    ps1 = cast(ops.PauliStringGateOperation, op1).pauli_string
    ps2 = cast(ops.PauliStringGateOperation, op2).pauli_string
    return protocols.commutes(ps1, ps2)


def pauli_string_dag_from_circuit(circuit: circuits.Circuit) -> circuits.CircuitDag:
    return circuits.CircuitDag.from_circuit(circuit, pauli_string_reorder_pred)

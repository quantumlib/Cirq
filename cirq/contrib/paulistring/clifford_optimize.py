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

from typing import Tuple, cast

from cirq import ops, circuits
from cirq.contrib.paulistring.pauli_string_phasor import (
    PauliStringPhasor)
from cirq.contrib.paulistring.convert_gate_set import (
    converted_gate_set)


def clifford_optimized_circuit(circuit: circuits.Circuit,
                               tolerance: float = 1e-8
                               ) -> circuits.Circuit:
    # Convert to a circuit with CliffordGates, CZs and other ignored gates
    c_cliff = converted_gate_set(circuit, no_clifford_gates=False,
                                 tolerance=tolerance)

    all_ops = list(c_cliff.all_operations())

    def find_merge_point(start_i: int,
                         string_op: PauliStringPhasor,
                         stop_at_cz: bool,
                         ) -> Tuple[int, PauliStringPhasor]:
        def continue_condition(op: ops.Operation,
                               current_string: PauliStringPhasor,
                               is_first: bool) -> int:
            if (isinstance(op, ops.GateOperation)
                and isinstance(op.gate, ops.CliffordGate)):
                return len(current_string.pauli_string) != 1
            if (isinstance(op, ops.GateOperation)
                and isinstance(op.gate, ops.Rot11Gate)):
                return not stop_at_cz
            if (isinstance(op, PauliStringPhasor)
                and len(op.qubits) == 1
                and (op.pauli_string[op.qubits[0]]
                     == current_string.pauli_string[op.qubits[0]])):
                return 2
            return False

        modified_op = string_op
        furthest_op = string_op
        furthest_i = start_i + 1
        i = start_i
        for i in range(start_i+1, len(all_ops)):
            op = all_ops[i]
            if not set(op.qubits) & set(modified_op.qubits):
                # No qubits in common
                continue
            cont_cond = continue_condition(op, modified_op, i == start_i+1)
            if not cont_cond:
                if len(modified_op.pauli_string) == 1:
                    furthest_op = modified_op
                    furthest_i = i
                break
            if cont_cond == 1:
                modified_op = modified_op.pass_operations_over(
                                    [op], after_to_before=True)
            if len(modified_op.pauli_string) == 1:
                furthest_op = modified_op
                furthest_i = i + 1

        return furthest_i, furthest_op

    def try_merge_clifford(cliff_op: ops.GateOperation, start_i: int) -> bool:
        orig_qubit, = cliff_op.qubits
        remaining_cliff_gate = ops.CliffordGate.I
        for pauli, quarter_turns in reversed(
                cast(ops.CliffordGate, cliff_op.gate).decompose_rotation()):
            trans = remaining_cliff_gate.transform(pauli)
            pauli = trans.to
            quarter_turns *= (1, -1)[trans.flip]
            string_op = PauliStringPhasor(
                ops.PauliString.from_single(cliff_op.qubits[0], pauli),
                half_turns=quarter_turns / 2)

            merge_i, merge_op = find_merge_point(start_i, string_op,
                                                 quarter_turns == 2)
            assert merge_i > start_i
            assert len(merge_op.pauli_string) == 1, 'PauliString length != 1'

            qubit, pauli = next(iter(merge_op.pauli_string.items()))
            quarter_turns = round(merge_op.half_turns * 2)
            quarter_turns *= (1, -1)[merge_op.pauli_string.negated]
            quarter_turns %= 4
            part_cliff_gate = ops.CliffordGate.from_quarter_turns(
                                        pauli, quarter_turns)

            other_op = all_ops[merge_i] if merge_i < len(all_ops) else None
            if other_op is not None and qubit not in set(other_op.qubits):
                other_op = None

            if (isinstance(other_op, ops.GateOperation)
                and isinstance(other_op.gate, ops.CliffordGate)):
                # Merge with another CliffordGate
                new_op = part_cliff_gate.merged_with(other_op.gate
                                                     )(qubit)
                all_ops[merge_i] = new_op
            elif (isinstance(other_op, ops.GateOperation)
                  and isinstance(other_op.gate, ops.Rot11Gate)
                  and other_op.gate.half_turns == 1
                  and quarter_turns == 2):
                # Pass whole Pauli gate over CZ, possibly adding a Z gate
                if pauli != ops.Pauli.Z:
                    other_qubit = other_op.qubits[
                                    other_op.qubits.index(qubit)-1]
                    all_ops.insert(merge_i+1,
                                   ops.CliffordGate.Z(other_qubit))
                all_ops.insert(merge_i+1, part_cliff_gate(qubit))
            elif isinstance(other_op, PauliStringPhasor):
                # Pass over a non-Clifford gate
                mod_op = other_op.pass_operations_over(
                                        [part_cliff_gate(qubit)])
                all_ops[merge_i] = mod_op
                all_ops.insert(merge_i+1, part_cliff_gate(qubit))
            elif merge_i > start_i + 1:
                # Moved Clifford through the circuit but nothing to merge
                all_ops.insert(merge_i, part_cliff_gate(qubit))
            else:
                # Couldn't move Clifford
                remaining_cliff_gate = remaining_cliff_gate.merged_with(
                                            part_cliff_gate)

        if remaining_cliff_gate == ops.CliffordGate.I:
            all_ops.pop(start_i)
            return True
        else:
            all_ops[start_i] = remaining_cliff_gate(orig_qubit)
            return False

    i = 0
    while i < len(all_ops):
        op = all_ops[i]
        if (isinstance(op, ops.GateOperation)
            and isinstance(op.gate, ops.CliffordGate)):
            if try_merge_clifford(op, i):
                i -= 1
        elif (isinstance(op, ops.GateOperation)
              and isinstance(op.gate, ops.Rot11Gate)
              and op.gate.half_turns == 1):
            for j in reversed(range(i)):
                op2 = all_ops[j]
                if not set(op.qubits) & set(op2.qubits):
                    # Don't share qubits
                    continue
                elif not (isinstance(op2, ops.GateOperation)
                          and isinstance(op2.gate, ops.Rot11Gate)
                          and op.gate.half_turns == 1):
                    # Not a CZ gate
                    break
                elif op == op2:
                    # Cancel two CZ gates
                    all_ops.pop(i)
                    all_ops.pop(j)
                    i -= 2
                    break
                else:
                    # Two CZ gates that share one qubit
                    # Pass through and keep looking
                    continue
        i += 1

    return circuits.Circuit.from_ops(
                all_ops,
                strategy=circuits.InsertStrategy.EARLIEST)

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

import json
import urllib.parse
from typing import List, cast, Tuple, Any, Iterable

from cirq import ops, circuits, devices, protocols
from cirq.contrib.quirk.linearize_circuit import linearize_circuit_qubits
from cirq.contrib.quirk.quirk_gate import (
    known_quirk_op_for_operation,
    QuirkOp,
    UNKNOWN_GATE,
    single_qubit_matrix_gate,
)


def _try_convert_to_quirk_gate(op: ops.Operation, prefer_unknown_gate_to_failure: bool) -> QuirkOp:
    quirk_gate = known_quirk_op_for_operation(op)
    if quirk_gate is not None:
        return quirk_gate
    matrix_op = single_qubit_matrix_gate(protocols.unitary(op, None))
    if matrix_op is not None:
        return matrix_op
    if prefer_unknown_gate_to_failure:
        return UNKNOWN_GATE
    raise TypeError(f'Unrecognized operation: {op!r}')


def _to_quirk_cols(
    op: ops.Operation, prefer_unknown_gate_to_failure: bool
) -> Iterable[Tuple[List[Any], bool]]:
    gate = _try_convert_to_quirk_gate(op, prefer_unknown_gate_to_failure)
    qubits = cast(Iterable[devices.LineQubit], op.qubits)

    max_index = max(q.x for q in qubits)
    col = [1] * (max_index + 1)
    for i, q in enumerate(qubits):
        col[q.x] = gate.keys[min(i, len(gate.keys) - 1)]
    yield col, gate.can_merge


def circuit_to_quirk_url(
    circuit: circuits.Circuit, prefer_unknown_gate_to_failure: bool = False, escape_url=True
) -> str:
    """Returns a Quirk URL for the given circuit.

    Args:
        circuit: The circuit to open in Quirk.
        prefer_unknown_gate_to_failure: If not set, gates that fail to convert
            will cause this function to raise an error. If set, a URL
            containing bad gates will be generated. (Quirk will open the URL,
            and replace the bad gates with parse errors, but still get the rest
            of the circuit.)
        escape_url: If set, the generated URL will have special characters such
            as quotes escaped using %. This makes it possible to paste the URL
            into forums and the command line and etc and have it properly
            parse. If not set, the generated URL will be more compact and human
            readable (and can still be pasted directly into a browser's address
            bar).

    Returns:

    """
    circuit = circuit.copy()
    linearize_circuit_qubits(circuit)

    cols: List[List[Any]] = []
    for moment in circuit:
        can_merges = []
        for op in moment.operations:
            for col, can_merge in _to_quirk_cols(op, prefer_unknown_gate_to_failure):
                if can_merge:
                    can_merges.append(col)
                else:
                    cols.append(col)
        if can_merges:
            merged_col = [1] * max(len(e) for e in can_merges)
            for col in can_merges:
                for i in range(len(col)):
                    if col[i] != 1:
                        merged_col[i] = col[i]
            cols.append(merged_col)

    circuit_json = json.JSONEncoder(
        ensure_ascii=False, separators=(',', ':'), sort_keys=True
    ).encode({'cols': cols})
    if escape_url:
        suffix = urllib.parse.quote(circuit_json)
    else:
        suffix = circuit_json
    return f'http://algassert.com/quirk#circuit={suffix}'

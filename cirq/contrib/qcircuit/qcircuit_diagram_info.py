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

from typing import Any, cast, Optional, Tuple

from cirq import ops, protocols

from cirq.contrib.qcircuit.qcircuit_diagrammable import (
    _FallbackQCircuitGate,
)


def _escape_text_for_latex(text):
    escaped = (text
               .replace('\\', '\\textbackslash{}')
               .replace('^', '\\textasciicircum{}')
               .replace('~', '\\textasciitilde{}')
               .replace('_', '\\_')
               .replace('{', '\\{')
               .replace('}', '\\}')
               .replace('$', '\\$')
               .replace('%', '\\%')
               .replace('&', '\\&')
               .replace('#', '\\#'))
    return '\\text{' + escaped + '}'


def get_multigate_parameters(gate: Any,
                              args: protocols.CircuitDiagramInfoArgs
                              ) -> Optional[Tuple[int, int]]:
    if ((not isinstance(gate, ops.InterchangeableQubitsGate)) or
        (args.qubit_map is None) or
        (args.known_qubits is None)):
        return None

    indices = [args.qubit_map[q] for q in args.known_qubits]
    min_index = min(indices)
    n_qubits = len(args.known_qubits)
    if sorted(indices) != list(range(min_index, min_index + n_qubits)):
        return None
    return min_index, n_qubits


def hardcoded_qcircuit_diagram_info(
        op: ops.Operation) -> Optional[protocols.CircuitDiagramInfo]:
    if not isinstance(op, ops.GateOperation):
        return None
    symbols = (
        (r'\targ',) if op.gate == ops.X else
        (r'\control', r'\control') if op.gate == ops.CZ else
        (r'\control', r'\targ') if op.gate == ops.CNOT else
        (r'\meter',) if ops.MeasurementGate.is_measurement(op.gate) else
        ())
    return (protocols.CircuitDiagramInfo(cast(Tuple[str, ...], symbols))
            if symbols else None)


def convert_text_diagram_info_to_qcircuit_diagram_info(
        info: protocols.CircuitDiagramInfo) -> protocols.CircuitDiagramInfo:
    labels = [_escape_text_for_latex(e) for e in info.wire_symbols]
    if info.exponent != 1:
        labels[0] += '^{' + str(info.exponent) + '}'
    symbols = tuple('\\gate{' + l + '}' for l in labels)
    return protocols.CircuitDiagramInfo(symbols)


def text_to_qcircuit_diagram_info(
        op: ops.Operation,
        args: protocols.CircuitDiagramInfoArgs
        ) -> Optional[protocols.CircuitDiagramInfo]:
    args = args.with_args(use_unicode_characters=False)
    info = protocols.circuit_diagram_info(op, args, default=None)
    if info is None:
        return None

    multigate_parameters = (
            get_multigate_parameters(op.gate, args)
            if isinstance(op, ops.GateOperation) else None)
    if multigate_parameters is None:
        return convert_text_diagram_info_to_qcircuit_diagram_info(info)

    min_index, n_qubits = multigate_parameters
    name = _escape_text_for_latex(
            str(cast(ops.GateOperation, op).gate).rsplit('**', 1)[0])
    if info.exponent != 1:
        name += '^{' + str(info.exponent) + '}'
    box = '\multigate{' + str(n_qubits - 1) + '}{' + name + '}'
    ghost = '\ghost{' + name + '}'
    assert args.qubit_map is not None
    assert args.known_qubits is not None
    symbols = tuple(box if (args.qubit_map[q] == min_index) else
                    ghost for q in args.known_qubits)
    return protocols.CircuitDiagramInfo(symbols,
                                        exponent=info.exponent,
                                        connected=False)


def get_qcircuit_diagram_info(op: ops.Operation,
                              args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
    info = hardcoded_qcircuit_diagram_info(op)
    if info is None:
        info = text_to_qcircuit_diagram_info(op, args)
    if info is None:
        if isinstance(op, ops.GateOperation):
            diagrammable = _FallbackQCircuitGate(op.gate)
        else:
            diagrammable = _FallbackQCircuitGate(op)
        info = diagrammable.qcircuit_diagram_info(args)
    return info

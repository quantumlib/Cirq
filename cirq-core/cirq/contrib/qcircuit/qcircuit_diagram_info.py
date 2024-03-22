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

from typing import Optional, Tuple

from cirq import ops, protocols


def escape_text_for_latex(text):
    escaped = (
        text.replace('\\', r'\textbackslash{}')
        .replace('{', r'\{')
        .replace('}', r'\}')
        .replace('^', r'\textasciicircum{}')
        .replace('~', r'\textasciitilde{}')
        .replace('_', r'\_')
        .replace('$', r'\$')
        .replace('%', r'\%')
        .replace('&', r'\&')
        .replace('#', r'\#')
    )
    return r'\text{' + escaped + '}'


def get_multigate_parameters(args: protocols.CircuitDiagramInfoArgs) -> Optional[Tuple[int, int]]:
    if (args.label_map is None) or (args.known_qubits is None):
        return None

    indices = [args.label_map[q] for q in args.known_qubits]
    min_index = min(indices)
    n_qubits = len(args.known_qubits)
    if sorted(indices) != list(range(min_index, min_index + n_qubits)):
        return None
    return min_index, n_qubits


def hardcoded_qcircuit_diagram_info(op: ops.Operation) -> Optional[protocols.CircuitDiagramInfo]:
    if not isinstance(op, ops.GateOperation):
        return None
    symbols = (
        (r'\targ',)
        if op.gate == ops.X
        else (
            (r'\control', r'\control')
            if op.gate == ops.CZ
            else (
                (r'\control', r'\targ')
                if op.gate == ops.CNOT
                else (r'\meter',) if isinstance(op.gate, ops.MeasurementGate) else ()
            )
        )
    )
    return protocols.CircuitDiagramInfo(symbols) if symbols else None


def convert_text_diagram_info_to_qcircuit_diagram_info(
    info: protocols.CircuitDiagramInfo,
) -> protocols.CircuitDiagramInfo:
    labels = [escape_text_for_latex(e) for e in info.wire_symbols]
    if info.exponent != 1:
        labels[0] += '^{' + str(info.exponent) + '}'
    symbols = tuple(r'\gate{' + l + '}' for l in labels)
    return protocols.CircuitDiagramInfo(symbols)


def multigate_qcircuit_diagram_info(
    op: ops.Operation, args: protocols.CircuitDiagramInfoArgs
) -> Optional[protocols.CircuitDiagramInfo]:
    if not (
        isinstance(op, ops.GateOperation) and isinstance(op.gate, ops.InterchangeableQubitsGate)
    ):
        return None

    multigate_parameters = get_multigate_parameters(args)
    if multigate_parameters is None:
        return None

    info = protocols.circuit_diagram_info(op, args, default=None)

    min_index, n_qubits = multigate_parameters
    name = escape_text_for_latex(
        str(op.gate).rsplit('**', 1)[0] if isinstance(op, ops.GateOperation) else str(op)
    )
    if (info is not None) and (info.exponent != 1):
        name += '^{' + str(info.exponent) + '}'
    box = r'\multigate{' + str(n_qubits - 1) + '}{' + name + '}'
    ghost = r'\ghost{' + name + '}'
    assert args.label_map is not None
    assert args.known_qubits is not None
    symbols = tuple(box if (args.label_map[q] == min_index) else ghost for q in args.known_qubits)
    # Force exponent=1 to defer to exponent formatting given above.
    return protocols.CircuitDiagramInfo(symbols, connected=False)


def fallback_qcircuit_diagram_info(
    op: ops.Operation, args: protocols.CircuitDiagramInfoArgs
) -> protocols.CircuitDiagramInfo:
    args = args.with_args(use_unicode_characters=False)
    info = protocols.circuit_diagram_info(op, args, default=None)
    if info is None:
        name = str(op.gate or op)
        n_qubits = len(op.qubits)
        symbols = tuple(f'#{i + 1}' if i else name for i in range(n_qubits))
        info = protocols.CircuitDiagramInfo(symbols)
    return convert_text_diagram_info_to_qcircuit_diagram_info(info)


def get_qcircuit_diagram_info(
    op: ops.Operation, args: protocols.CircuitDiagramInfoArgs
) -> protocols.CircuitDiagramInfo:
    info = hardcoded_qcircuit_diagram_info(op)
    if info is None:
        info = multigate_qcircuit_diagram_info(op, args)
    if info is None:
        info = fallback_qcircuit_diagram_info(op, args)
    return info

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

from typing import Any, Callable, Optional

from cirq import abc, circuits, extension, ops
from cirq.contrib.qcircuit_diagrammable_gate import (
    QCircuitDiagrammableGate,
    fallback_qcircuit_extensions,
)


class _QCircuitQubit(ops.QubitId):
    def __init__(self, sub: ops.QubitId):
        self.sub = sub

    def __str__(self):
        # TODO: If qubit name ends with digits, turn them into subscripts.
        return '\\lstick{\\text{' + str(self.sub) + '}}&'

    def __eq__(self, other):
        if not isinstance(other, _QCircuitQubit):
            return NotImplemented
        return self.sub == other.sub

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((_QCircuitQubit, self.sub))


class _QCircuitGate(ops.TextDiagrammableGate, metaclass=abc.ABCMeta):
    def __init__(self, sub: QCircuitDiagrammableGate):
        self.sub = sub

    def text_diagram_exponent(self):
        return 1

    def text_diagram_wire_symbols(self,
                                  qubit_count: Optional[int] = None,
                                  use_unicode_characters: bool = True):
        return self.sub.qcircuit_wire_symbols(qubit_count)


def _render(diagram: circuits.TextDiagramDrawer) -> str:
    w = diagram.width()
    h = diagram.height()

    qwx = {(x, y + 1)
           for x, y1, y2 in diagram.vertical_lines
           for y in range(y1, y2)}

    qw = {(x, y)
          for y, x1, x2 in diagram.horizontal_lines
          for x in range(x1, x2)}

    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            cell = []
            key = (x, y)
            v = diagram.entries.get(key)
            if v is not None:
                cell.append(' ' + v + ' ')
            if key in qw:
                cell.append('\\qw ')
            if key in qwx:
                cell.append('\\qwx ')
            row.append(''.join(cell))
        rows.append('&'.join(row) + '\qw')

    grid = '\\\\\n'.join(rows)

    output = '\Qcircuit @R=1em @C=0.75em { \\\\ \n' + grid + ' \\\\ \n \\\\ }'

    return output


def _wrap_operation(op: ops.Operation,
                    ext: extension.Extensions) -> ops.Operation:
    new_qubits = [_QCircuitQubit(e) for e in op.qubits]
    new_gate = ext.try_cast(op.gate, QCircuitDiagrammableGate)
    if new_gate is None:
        new_gate = fallback_qcircuit_extensions.cast(op.gate,
                                                     QCircuitDiagrammableGate)
    return ops.Operation(_QCircuitGate(new_gate), new_qubits)


def _wrap_moment(moment: circuits.Moment,
                 ext: extension.Extensions) -> circuits.Moment:
    return circuits.Moment(_wrap_operation(op, ext)
                           for op in moment.operations)


def _wrap_circuit(circuit: circuits.Circuit,
                  ext: extension.Extensions) -> circuits.Circuit:
    return circuits.Circuit(_wrap_moment(moment, ext)
                            for moment in circuit.moments)


def circuit_to_latex_using_qcircuit(
        circuit: circuits.Circuit,
        ext: extension.Extensions = None,
        qubit_order_key: Callable[[ops.QubitId], Any] = None) -> str:
    """Returns a QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to represent in latex.
        ext: Extensions used when attempting to cast gates into
            QCircuitDiagrammableGate instances (before falling back to the
            default wrapping methods).
        qubit_order_key: Determines the order of qubit wires in the diagram.

    Returns:
        Latex code for the diagram.
    """
    if ext is None:
        ext = extension.Extensions()
    qcircuit = _wrap_circuit(circuit, ext)
    diagram = qcircuit.to_text_diagram_drawer(
        ext,
        qubit_name_suffix='',
        qubit_order_key=(None
                         if qubit_order_key is None
                         else lambda e: qubit_order_key(e.sub)))
    return _render(diagram)

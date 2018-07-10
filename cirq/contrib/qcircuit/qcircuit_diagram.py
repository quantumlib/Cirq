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

from cirq import circuits, extension, ops
from cirq.contrib.qcircuit.qcircuit_diagrammable_gate import (
    QCircuitDiagrammableGate,
    fallback_qcircuit_extensions,
)


class _QCircuitQubit(ops.QubitId):
    def __init__(self, sub: ops.QubitId) -> None:
        self.sub = sub

    def __repr__(self):
        return '_QCircuitQubit({!r})'.format(self.sub)

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


class _QCircuitGate(ops.Gate, ops.TextDiagrammable):
    def __init__(self, sub: QCircuitDiagrammableGate) -> None:
        self.sub = sub

    def text_diagram_info(self, args: ops.TextDiagramInfoArgs
                          ) -> ops.TextDiagramInfo:
        return ops.TextDiagramInfo(
            wire_symbols=self.sub.qcircuit_wire_symbols(args.known_qubit_count))


def _render(diagram: circuits.TextDiagramDrawer) -> str:
    w = diagram.width()
    h = diagram.height()

    qwx = {(x, y + 1)
           for x, y1, y2, _ in diagram.vertical_lines
           for y in range(y1, y2)}

    qw = {(x, y)
          for y, x1, x2, _ in diagram.horizontal_lines
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
    new_gate = ext.try_cast(QCircuitDiagrammableGate, op.gate)
    if new_gate is None:
        new_gate = fallback_qcircuit_extensions.cast(QCircuitDiagrammableGate,
                                                     op.gate)
    return ops.GateOperation(_QCircuitGate(new_gate), new_qubits)


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
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT) -> str:
    """Returns a QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to represent in latex.
        ext: Extensions used when attempting to cast gates into
            QCircuitDiagrammableGate instances (before falling back to the
            default wrapping methods).
        qubit_order: Determines the order of qubit wires in the diagram.

    Returns:
        Latex code for the diagram.
    """
    if ext is None:
        ext = extension.Extensions()
    qcircuit = _wrap_circuit(circuit, ext)

    # Note: can't be a lambda because we need the type hint.
    def get_sub(q: _QCircuitQubit) -> ops.QubitId:
        return q.sub

    diagram = qcircuit.to_text_diagram_drawer(
        ext,
        qubit_name_suffix='',
        qubit_order=ops.QubitOrder.as_qubit_order(qubit_order).map(
            internalize=get_sub, externalize=_QCircuitQubit))
    return _render(diagram)

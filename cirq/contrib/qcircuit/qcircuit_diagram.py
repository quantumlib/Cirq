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
from cirq.contrib.qcircuit.qcircuit_diagrammable import (
    QCircuitDiagrammable,
    known_qcircuit_operation_symbols,
    _TextToQCircuitDiagrammable,
    _FallbackQCircuitGate,
)


def qcircuit_qubit_namer(qubit: ops.QubitId):
    # TODO: If qubit name ends with digits, turn them into subscripts.
    return '\\lstick{\\text{' + str(qubit) + '}}&'


class _QCircuitOperation(ops.Operation):
    def __init__(self,
                 sub_operation: ops.Operation,
                 diagrammable: QCircuitDiagrammable) -> None:
        self.sub_operation = sub_operation
        self.diagrammable = diagrammable

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return self.diagrammable.qcircuit_diagram_info(args)

    @property
    def qubits(self):
        return self.sub_operation.qubits

    def with_qubits(self, *new_qubits: ops.QubitId) -> '_QCircuitOperation':
        return _QCircuitOperation(
            self.sub_operation.with_qubits(*new_qubits),
            self.diagrammable)


def _render(diagram: circuits.TextDiagramDrawer) -> str:
    w = diagram.width()
    h = diagram.height()

    qwx = {(x, y + 1)
           for x, y1, y2, _ in diagram.vertical_lines
           for y in range(y1, y2)}

    qw = {(x, y)
          for y, x1, x2, _ in diagram.horizontal_lines
          for x in range(x1, x2)}

    diagram2 = circuits.TextDiagramDrawer()
    for y in range(h):
        for x in range(max(0, w - 1)):
            key = (x, y)
            diagram_text = diagram.entries.get(key)
            v = '&' + (diagram_text.text if diagram_text else  '') + ' '
            diagram2.write(2*x + 1, y, v)
            post1 = '\\qw' if key in qw else ''
            post2 = '\\qwx' if key in qwx else ''
            diagram2.write(2*x + 2, y, post1 + post2)
        diagram2.write(2*w - 1, y, '&\\qw\\\\')
    grid = diagram2.render(horizontal_spacing=0, vertical_spacing=0)

    output = '\Qcircuit @R=1em @C=0.75em {\n \\\\\n' + grid + '\n \\\\\n}'

    return output


def _wrap_operation(op: ops.Operation) -> ops.Operation:
    diagrammable = known_qcircuit_operation_symbols(op)
    if diagrammable is None:
        info = protocols.circuit_diagram_info(op, default=None)
        if info is not None:
            diagrammable = _TextToQCircuitDiagrammable(
                cast(protocols.SupportsCircuitDiagramInfo, op))
        elif isinstance(op, ops.GateOperation):
            diagrammable = _FallbackQCircuitGate(op.gate)
        else:
            diagrammable = _FallbackQCircuitGate(op)
    return _QCircuitOperation(op, diagrammable)


def get_qcircuit_diagram_info(op: ops.Operation,
                              args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
    return cast(protocols.CircuitDiagramInfo,
            cast(protocols.SupportsCircuitDiagramInfo, _wrap_operation(op)
                )._circuit_diagram_info_(args))


def circuit_to_latex_using_qcircuit(
        circuit: circuits.Circuit,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT) -> str:
    """Returns a QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to represent in latex.
        qubit_order: Determines the order of qubit wires in the diagram.

    Returns:
        Latex code for the diagram.
    """
    diagram = circuit.to_text_diagram_drawer(
        qubit_namer=qcircuit_qubit_namer,
        qubit_order=qubit_order,
        get_circuit_diagram_info=get_qcircuit_diagram_info)
    return _render(diagram)

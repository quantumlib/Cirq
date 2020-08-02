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
from typing import TYPE_CHECKING

from cirq import circuits, ops
from cirq.contrib.qcircuit.qcircuit_diagram_info import (
    escape_text_for_latex, get_qcircuit_diagram_info)

if TYPE_CHECKING:
    import cirq


def qcircuit_qubit_namer(qubit: 'cirq.Qid') -> str:
    """Returns the latex code for a QCircuit label of given qubit.

        Args:
            qubit: The qubit which name to represent.

        Returns:
            Latex code for the label.
    """
    return r'\lstick{' + escape_text_for_latex(str(qubit)) + '}&'


def _render(diagram: circuits.TextDiagramDrawer) -> str:
    w = diagram.width()
    h = diagram.height()

    # potential places to place \qwx and \qw
    qwx = {(x, y + 1)
           for x, y1, y2, _ in diagram.vertical_lines
           for y in range(int(y1), int(y2))}

    qw = {(x, y)
          for y, x1, x2, _ in diagram.horizontal_lines
          for x in range(int(x1), int(x2))}

    diagram2 = circuits.TextDiagramDrawer()
    for y in range(h):
        # have a none wire item on this row
        row_has_item = False
        # current horizontal line has something that need to be connected
        for x in range(max(0, w - 1)):
            key = (x, y)
            diagram_text = diagram.entries.get(key)
            v = '&' + (diagram_text.text if diagram_text else '') + ' '
            if diagram_text:
                row_has_item = True
            diagram2.write(2*x + 1, y, v)
            # if has none wire before, should draw wire
            post1 = r'\qw' if key in qw and row_has_item else ''
            post2 = ''
            if key in qwx:
                # see if the item in row below can be connected by line
                lower_item = diagram.entries.get((x, y + 1))
                # see if the upper row needs to be connected
                upper_wire = diagram2.entries.get((2 * x + 2, y - 1))
                upper_wire_t = (upper_wire.text
                                if upper_wire is not None else '')
                upper_item = diagram2.entries.get((2 * x + 1, y - 1))
                upper_item_t = (upper_item.text if
                                upper_item is not None else '')
                req_qwx = (r'\qwx' in upper_wire_t or
                           r'\control' in upper_item_t)
                if lower_item is not None or req_qwx:
                    post2 = r'\qwx'
            diagram2.write(2*x + 2, y, post1 + post2)
        # end also needs to be different whether the row has item
        if row_has_item:
            diagram2.write(2*w - 1, y, r'&\qw\\')
        else:
            diagram2.write(2*w - 1, y, r'& \\')  # coverage: ignore
    grid = diagram2.render(horizontal_spacing=0, vertical_spacing=0)

    output = '\\Qcircuit @R=1em @C=0.75em {\n \\\\\n' + grid + '\n \\\\\n}'

    return output


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

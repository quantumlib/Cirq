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

import re
from cirq import circuits, ops
from cirq.contrib.qcircuit.qcircuit_diagram_info import (
    get_qcircuit_diagram_info)


def qcircuit_qubit_namer(qubit: ops.Qid) -> str:
    """Returns the latex code for a QCircuit label of given qubit.

    If the qubit name ends with digits, those digits will be subsripted.
    Note, if the name is only made up of digits, the subscript will not be
    applied.

        Args:
            qubit: The qubit which name to represent.

        Returns:
            Latex code for the label.
    """
    name = re.sub(r'(?<=\D)(\d+)$', r'$_{\1}$', str(qubit))
    return r'\lstick{\text{' + name + '}}&'


def _render(diagram: circuits.TextDiagramDrawer) -> str:
    w = diagram.width()
    h = diagram.height()

    qwx = {(x, y + 1)
           for x, y1, y2, _ in diagram.vertical_lines
           for y in range(int(y1), int(y2))}

    qw = {(x, y)
          for y, x1, x2, _ in diagram.horizontal_lines
          for x in range(int(x1), int(x2))}

    diagram2 = circuits.TextDiagramDrawer()
    for y in range(h):
        for x in range(max(0, w - 1)):
            key = (x, y)
            diagram_text = diagram.entries.get(key)
            v = '&' + (diagram_text.text if diagram_text else  '') + ' '
            diagram2.write(2*x + 1, y, v)
            post1 = r'\qw' if key in qw else ''
            post2 = r'\qwx' if key in qwx else ''
            diagram2.write(2*x + 2, y, post1 + post2)
        diagram2.write(2*w - 1, y, r'&\qw\\')
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

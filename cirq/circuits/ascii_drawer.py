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

"""Utility methods for converting circuits to/from ascii diagram."""

from typing import Dict, List, Callable, Any

from cirq import ops
from cirq.circuits.circuit import Circuit
from cirq.circuits.moment import Moment
from cirq.extension.extensions import Extensions


class _AsciiDiagramDrawer:

    def __init__(self):
        self.entries = dict()
        self.vertical_lines = []
        self.horizontal_lines = []

    def write(self, x: int, y: int, text: str):
        """Adds text to the given location."""
        if (x, y) in self.entries:
            self.entries[(x, y)] += text
        else:
            self.entries[(x, y)] = text

    def content_present(self, x: int, y: int) -> bool:
        """Determines if a line or printed text is at the given location."""

        # Text?
        if (x, y) in self.entries:
            return True

        # Vertical line?
        if any(line_x == x and y1 < y < y2
               for line_x, y1, y2 in self.vertical_lines):
            return True

        # Horizontal line?
        if any(line_y == y and x1 < x < x2
               for line_y, x1, x2 in self.horizontal_lines):
            return True

        return False

    def vertical_line(self, x: int, y1: int, y2: int):
        """Adds a line from (x, y1) to (x, y2)."""
        y1, y2 = sorted([y1, y2])
        self.vertical_lines.append((x, y1, y2))

    def horizontal_line(self, y, x1, x2):
        """Adds a line from (x1, y) to (x2, y)."""
        x1, x2 = sorted([x1, x2])
        self.horizontal_lines.append((y, x1, x2))

    def transpose(self):
        """Returns the same diagram, but mirrored across its diagonal."""
        out = _AsciiDiagramDrawer()
        out.entries = {(y, x): v for (x, y), v in self.entries.items()}
        out.vertical_lines = list(self.horizontal_lines)
        out.horizontal_lines = list(self.vertical_lines)
        return out

    def width(self):
        """Determines how many entry columns are in the diagram."""
        max_x = -1
        for x, _ in self.entries.keys():
            max_x = max(max_x, x)
        for x, _, _ in self.vertical_lines:
            max_x = max(max_x, x)
        for _, x1, x2 in self.horizontal_lines:
            max_x = max(max_x, x1, x2)
        return 1 + max_x

    def height(self):
        """Determines how many entry rows are in the diagram."""
        max_y = -1
        for _, y in self.entries.keys():
            max_y = max(max_y, y)
        for y, _, _ in self.horizontal_lines:
            max_y = max(max_y, y)
        for _, y1, y2 in self.vertical_lines:
            max_y = max(max_y, y1, y2)
        return 1 + max_y

    def render(self,
               horizontal_spacing: int = 1,
               vertical_spacing: int = 1,
               crossing_char: str = None,
               use_unicode_characters: bool = False) -> str:
        """Outputs ascii text containing the ascii diagram."""

        pipe = '│' if use_unicode_characters else '|'
        dash = '─' if use_unicode_characters else '-'
        if crossing_char is None:
            crossing_char = '┼' if use_unicode_characters else '+'

        dx = 1 + horizontal_spacing
        dy = 1 + vertical_spacing
        w = self.width() * dx - horizontal_spacing
        h = self.height() * dy - vertical_spacing

        grid = [[''] * w for _ in range(h)]
        extend_char = [[' '] * w for _ in range(h)]

        for x, y1, y2 in self.vertical_lines:
            x *= dx
            y1 *= dy
            y2 *= dy
            for y in range(y1, y2):
                grid[y][x] = pipe

        for y, x1, x2 in self.horizontal_lines:
            y *= dy
            x1 *= dx
            x2 *= dx
            for x in range(x1, x2):
                if grid[y][x] == pipe:
                    grid[y][x] = crossing_char
                else:
                    grid[y][x] = dash
                extend_char[y][x] = dash

        for (x, y), v in self.entries.items():
            x *= dx
            y *= dy
            grid[y][x] = v

        for col in range(w):
            col_width = max(1, max(len(grid[y][col]) for y in range(h)))
            for row in range(h):
                missing = col_width - len(grid[row][col])
                grid[row][col] += extend_char[row][col] * missing

        return '\n'.join(''.join(row).rstrip() for row in grid)


def _get_operation_symbols(op: ops.Operation, ext: Extensions) -> List[str]:
    ascii_gate = ext.try_cast(op.gate, ops.AsciiDiagrammableGate)
    if ascii_gate is not None:
        return ascii_gate.ascii_wire_symbols()
    name = repr(op.gate)
    if len(op.qubits) == 1:
        return [name]
    return ['{}:{}'.format(name, i) for i in range(len(op.qubits))]


def _get_operation_exponent(op: ops.Operation, ext: Extensions) -> float:
    ascii_gate = ext.try_cast(op.gate, ops.AsciiDiagrammableGate)
    if ascii_gate is not None:
        return ascii_gate.ascii_exponent()
    return 1


def _to_ascii_moment(moment: Moment,
                     ext: Extensions,
                     qubit_map: Dict[ops.QubitId, int],
                     out_diagram: _AsciiDiagramDrawer):
    if not moment.operations:
        return []

    x0 = out_diagram.width()
    for op in moment.operations:
        indices = [qubit_map[q] for q in op.qubits]
        y1 = min(indices)
        y2 = max(indices)

        # Find an available column.
        x = x0
        while any(out_diagram.content_present(x, y)
                  for y in range(y1, y2 + 1)):
            x += 1

        # Draw vertical line linking the gate's qubits.
        if y2 > y1:
            out_diagram.vertical_line(x, y1, y2)

        # Print gate qubit labels.
        symbols = _get_operation_symbols(op, ext)
        for s, q in zip(symbols, op.qubits):
            out_diagram.write(x, qubit_map[q], s)

        # Add an exponent to the first label.
        exponent = _get_operation_exponent(op, ext)
        if exponent != 1:
            out_diagram.write(x, y1, '^' + repr(exponent))


def _str_lexi(value):
    """0-pads digits in a string to hack int order into lexicographic order."""
    s = str(value)

    was_on_digits = False
    last_transition = 0
    output = []

    def dump(k):
        chunk = s[last_transition:k]
        if was_on_digits:
            chunk = chunk.rjust(8, '0')
        output.append(chunk)

    for i in range(len(s)):
        on_digits = s[i].isdigit()
        if was_on_digits != on_digits:
            dump(i)
            was_on_digits = on_digits
            last_transition = i

    dump(len(s))
    return ''.join(output)


def to_ascii(circuit: Circuit,
             ext: Extensions = Extensions(),
             use_unicode_characters: bool = False,
             transpose: bool = False,
             qubit_order_key: Callable[[ops.QubitId], Any] = None) -> str:
    """Paints an ascii diagram describing the given circuit.

    Args:
        circuit: The circuit to turn into a diagram.
        ext: For extending gates to implement AsciiDiagrammableGate.
        use_unicode_characters: Activates the use of box-drawing characters.
        transpose: Arranges the wires vertically instead of horizontally.
        qubit_order_key: Transforms each qubit into a key that determines how
            the qubits are ordered in the diagram. Qubits with lower keys come
            first. Defaults to the qubit's __str__, but augmented so that
            lexicographic ordering will respect the order of integers within
            the string (e.g. "name10" will come after "name2").

    Returns:
        The ascii diagram.

    Raises:
        ValueError: The circuit contains gates that don't support ascii
            diagramming.
    """
    if qubit_order_key is None:
        qubit_order_key = _str_lexi

    qubits = {
        q
        for moment in circuit.moments for op in moment.operations
        for q in op.qubits
    }
    ordered_qubits = sorted(qubits, key=qubit_order_key)
    qubit_map = {ordered_qubits[i]: i for i in range(len(ordered_qubits))}

    diagram = _AsciiDiagramDrawer()
    for q, i in qubit_map.items():
        diagram.write(0, i, str(q) + ('' if transpose else ': '))

    for moment in [Moment()] * 2 + circuit.moments + [Moment()]:
        _to_ascii_moment(moment, ext, qubit_map, diagram)

    w = diagram.width()
    for i in qubit_map.values():
        diagram.horizontal_line(i, 0, w)

    if transpose:
        return diagram.transpose().render(
            crossing_char='─' if use_unicode_characters else '-',
            use_unicode_characters=use_unicode_characters)
    return diagram.render(
        crossing_char='┼' if use_unicode_characters else '|',
        horizontal_spacing=3,
        use_unicode_characters=use_unicode_characters)

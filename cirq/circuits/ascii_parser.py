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

from typing import Optional, List, Dict, Iterator, Tuple, Callable

from cirq import ops
from cirq.circuits.circuit import Circuit
from cirq.circuits.moment import Moment

# Characters that indicate nothing is happening here in the circuit.
_ENTRY_TERMINATOR_CHARS = {' ', '-', '|', '+'}


def _wire_name_to_qubit(qubit_name_parser: Callable[[str],
                                                    Optional[ops.QubitId]],
                        name: str) -> ops.QubitId:
    q = qubit_name_parser(name)
    if q is None:
        raise NotImplementedError('Bad name: {}'.format(repr(name)))
    return q


def _column_entry_is_linky(entry):
    return entry[0] in {'|', '+', '.', '@'}


def _gate_name_to_gate(gate_name: str) -> Optional[ops.Gate]:
    """Determines the gate that corresponds to some diagram text."""

    if not gate_name or gate_name in _ENTRY_TERMINATOR_CHARS:
        return None

    if gate_name in ['z', 'Z', '.', '@']:
        return ops.Z

    if gate_name in ['t', 'T']:
        return ops.Z**0.25

    if gate_name in ['s', 'S']:
        return ops.Z**0.5

    if gate_name in ['x', 'X']:
        return ops.X

    if gate_name in ['y', 'Y']:
        return ops.Y

    if gate_name in ['h', 'H']:
        return ops.H

    if gate_name in ['m', 'M']:
        return ops.MeasurementGate()

    raise ValueError('Unrecognized gate_name: {}'.format(repr(gate_name)))


def _col_entry_to_gate_and_exponent(
        entry: str) -> Tuple[Optional[ops.Gate], float]:
    if '^' in entry:
        gate_name, exponent_str = entry.split('^')
        exponent = float(exponent_str)
    else:
        gate_name = entry
        exponent = 1

    return _gate_name_to_gate(gate_name), exponent


def _column_entries_to_interaction_groups(
    column_entries: List[str]
) -> Iterator[Tuple[Dict[int, ops.Gate], float]]:
    """Figures out the groups of linked operations in a column.

  Args:
    column_entries: The text entries from each row starting at this column.

  Yields:
    Linked groups of gates, along with their exponent modifier.
  """
    linkers = [_column_entry_is_linky(c) for c in column_entries]
    is_linked_to_next = [a or b for a, b in zip(linkers[1:], linkers)] + [
        False]

    # Scan over column, yielding linked groups of operations.
    linked_ops = {}
    linked_exponent = 1
    for wire in range(len(column_entries)):
        gate, exponent = _col_entry_to_gate_and_exponent(column_entries[wire])
        linked_exponent *= exponent
        if gate is not None:
            linked_ops[wire] = gate

        # Note: triggers correctly on last line.
        if linked_ops and not is_linked_to_next[wire]:
            yield linked_ops, linked_exponent
            linked_ops = {}


def _interaction_group_to_operation(
        group: Dict[int, ops.Gate], exponent: float,
        qubit_map: Dict[int, ops.QubitId]) -> ops.Operation:
    """Converts linked gates from a diagram into a full operation.

    Args:
        group: The linked gates and the wires they were applied to.
        exponent: The product of all the modifiers applied to the linked gates.
        qubit_map: Which lines correspond to which qubits.

    Returns:
        The operation the linked gates correspond to.
    """
    if len(group) == 1:
        wire = list(group.keys())[0]
        gate = group[wire]
        target = qubit_map[wire]
        if exponent != 1:
            gate **= exponent
        return gate(target)

    if len(group) == 2:
        wire1, wire2 = list(group.keys())
        q1, q2 = qubit_map[wire1], qubit_map[wire2]
        gate1, gate2 = group[wire1], group[wire2]

        if gate1 == ops.Z and gate2 == ops.Z:
            gate = ops.CZ
        elif gate1 == ops.X and gate2 == ops.Z:
            gate = ops.CNOT
            q1, q2 = q2, q1
        elif gate1 == ops.Z and gate2 == ops.X:
            gate = ops.CNOT
        else:
            raise NotImplementedError('2-qubit operation: {}'.format(group))

        if exponent != 1:
            gate **= exponent
        return gate(q1, q2)

    raise NotImplementedError('k-qubit interaction: {}'.format(group))


def _snip_column_entries(lines: List[str], col: int,
                         qubit_map: Dict[int, ops.QubitId]) -> List[str]:
    """Determines the text entries in a column of an ascii diagram.

    Args:
        lines: The circuit lines to query and modify.
        col: The index of the column.
        qubit_map: Which lines correspond to which qubits.

    Mutates:
        lines: Replaces the column's entries with wire/space characters.

    Returns:
        A line-by-line list of the entries extracted from the column.
  """

    entries = []
    for row in range(len(lines)):
        line = lines[row]

        n = 0
        # Exponents can follow terminators in the first column.
        if col + 1 < len(line) and line[col + 1] == '^':
            n = 2
        # Scan ahead until the term terminates.
        while col + n < len(line) and line[
                    col + n] not in _ENTRY_TERMINATOR_CHARS:
            n += 1
        # Must include at least the leading character (to track control lines).
        n = max(n, 1)

        filler = '-' if row in qubit_map else ' '
        lines[row] = line[:col] + filler * n + line[col + n:]
        entry = line[col:col + n]
        entries.append(entry)
    return entries


def _snip_column_ops(lines: List[str], col: int,
                     qubit_map: Dict[int, ops.QubitId]) -> List[ops.Operation]:
    """Determines the operations in a column of an ascii diagram.

    Args:
        lines: The circuit lines to query and modify.
        col: The index of the column.
        qubit_map: Which lines correspond to which qubits.

    Mutates:
        lines: Replaces the column's entries with wire/space characters.

    Yields:
        Operations from the column.
  """

    col_entries = _snip_column_entries(lines, col, qubit_map)
    interaction_groups = _column_entries_to_interaction_groups(col_entries)

    for group, exponent in interaction_groups:
        yield _interaction_group_to_operation(group, exponent, qubit_map)


def _snip_qubit_map(lines: List[str],
                    qubit_name_parser: Callable[[str],
                                                Optional[ops.QubitId]]
                    ) -> Dict[int, ops.QubitId]:
    """Determines which wires are qubits.

    Args:
        lines: The circuit lines to query and modify.

    Mutates:
        lines: Replaces qubit labels with wire characters.

    Returns:
        A map from line indices to qubit ids.

    Raises:
        ValueError: Invalid qubit names.
    """
    is_wire = [line and not line.startswith(' ') for line in lines]
    wires = [i for i in range(len(lines)) if is_wire[i]]
    wire_lines = [lines[wire] for wire in wires]
    names = [line.split(':')[0] for line in wire_lines]
    qubits = [_wire_name_to_qubit(qubit_name_parser, name) for name in names]
    wire_to_qubit = {w: q for w, q in zip(wires, qubits)}

    # Check for duplicates.
    qubit_to_wire = {q: w for q, w in zip(qubits, wires)}
    for w, q in zip(wires, qubits):
        if qubit_to_wire[q] != w:
            raise ValueError('Duplicate qubit: {}'.format(q))

    # Erase names.
    for wi in wires:
        n = lines[wi].index(':') + 1
        while lines[wi][n] == ' ':
            n += 1
        lines[wi] = '-' * n + lines[wi][n:]

    return wire_to_qubit


def from_ascii(text: str,
               qubit_name_parser: Callable[[str],
                                           Optional[ops.QubitId]]) -> Circuit:
    """Parses a Circuit out of an ascii diagram.

    Args:
        text: The text containing the ascii diagram.
        qubit_name_parser: Understands how to make sense of the qubit names.

    Returns:
        The parsed circuit.

    Raises:
        ValueError: The ascii diagram isn't valid.
    """
    lines = text.split('\n')
    lines = [line for line in lines if not line.startswith('#')]
    w = max(len(s) for s in lines)
    lines = [line.ljust(w, ' ') for line in lines]

    qubit_map = _snip_qubit_map(lines, qubit_name_parser)

    circuit = Circuit()
    for col in range(w):
        col_ops = list(_snip_column_ops(lines, col, qubit_map))
        if col_ops:
            circuit.moments.append(Moment(col_ops))
    return circuit

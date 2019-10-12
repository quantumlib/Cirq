# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import urllib.parse
from typing import Any, List, Dict, Optional, Sequence, cast, TYPE_CHECKING, \
    Iterable

from cirq import devices, circuits, ops
from cirq.contrib.quirk.cells import (
    Cell,
    CellMaker,
    CellMakerArgs,
    generate_all_quirk_cell_makers,
    ExplicitOperationsCell,
)

if TYPE_CHECKING:
    import cirq


def quirk_url_to_circuit(
        quirk_url: str,
        *,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        extra_cell_makers: Iterable['cirq.contrib.quirk.cells.CellMaker'] = ()
) -> 'cirq.Circuit':
    """Parses a Cirq circuit out of a Quirk URL.

    Args:
        quirk_url: The URL of a bookmarked Quirk circuit. It is not required
            that the domain be "algassert.com/quirk". The only important part of
            the URL is the fragment (the part after the #).
        qubits: Qubits to use in the circuit. The length of the list must be
            at least the number of qubits in the Quirk circuit (including unused
            qubits). The maximum number of qubits in a Quirk circuit is 16.
            This argument defaults to `cirq.LineQubit.range(16)` when not
            specified.
        extra_cell_makers: A list of non-standard Quirk cell makers. This can be
            used to parse URLs that come from a modified version of Quirk that
            includes gates that Quirk doesn't define. See
            `cirq.contrib.quirk.cells.CellMaker`.

    Examples:
        >>> print(cirq.contrib.quirk.quirk_url_to_circuit(
        ...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}'
        ... ))
        0: ───H───@───
                  │
        1: ───────X───

        >>> print(cirq.contrib.quirk.quirk_url_to_circuit(
        ...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
        ...     qubits=[cirq.NamedQubit('Alice'), cirq.NamedQubit('Bob')]
        ... ))
        Alice: ───H───@───
                      │
        Bob: ─────────X───

        >>> print(cirq.contrib.quirk.quirk_url_to_circuit(
        ...     'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
        ...     extra_cell_makers=[
        ...         cirq.contrib.quirk.cells.CellMaker(
        ...             identifier='iswap',
        ...             size=2,
        ...             maker=lambda args: cirq.ISWAP(*args.qubits))
        ...     ]))
        0: ───iSwap───
              │
        1: ───iSwap───

    Returns:
        The parsed circuit.
    """

    parsed_url = urllib.parse.urlparse(quirk_url)
    if not parsed_url.fragment:
        return circuits.Circuit()

    if not parsed_url.fragment.startswith('circuit='):
        raise ValueError('Not a valid quirk url. The URL fragment (the part '
                         'after the #) must start with "circuit=".\n'
                         f'URL={quirk_url}')

    # URL parser may not have fixed escaped characters in the fragment.
    json_text = parsed_url.fragment[len('circuit='):]
    if '%22' in json_text:
        json_text = urllib.parse.unquote(json_text)

    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError('Circuit JSON must have a top-level dictionary.\n'
                         f'URL={quirk_url}')
    if not data.keys() <= {'cols', 'gates', 'init'}:
        raise ValueError(f'Unrecognized Circuit JSON keys.\nURL={quirk_url}')
    if 'gates' in data:
        raise NotImplementedError('Custom gates not supported yet.\n'
                                  f'URL={quirk_url}')
    if 'init' in data:
        raise NotImplementedError('Custom initial states not supported yet.\n'
                                  f'URL={quirk_url}')
    if 'cols' not in data:
        raise ValueError('Circuit JSON dictionary must have a "cols" entry.\n'
                         f'URL={quirk_url}')

    cols = data['cols']
    if not isinstance(cols, list):
        raise ValueError('Circuit JSON cols must be a list.\n'
                         f'URL={quirk_url}')

    # Parse column json into cells.
    registry = {
        entry.identifier: entry
        for entry in [*generate_all_quirk_cell_makers(), *extra_cell_makers]
    }
    parsed_cols: List[List[Optional[Cell]]] = []
    for i, col in enumerate(cols):
        parsed_cols.append(_parse_col_cells(registry, i, col))

    # Apply column modifiers (controls and inputs).
    for col in parsed_cols:
        for i in range(len(col)):
            cell = col[i]
            if cell is not None:
                cell.modify_column(col)

    # Apply persistent modifiers (classical assignments).
    persistent_mods = {}
    for c in parsed_cols:
        for cell in c:
            if cell is not None:
                for key, modifier in cell.persistent_modifiers().items():
                    persistent_mods[key] = modifier
        for i in range(len(c)):
            for modifier in persistent_mods.values():
                cell = c[i]
                if cell is not None:
                    c[i] = modifier(cell)

    # Extract circuit operations from modified cells.
    result = circuits.Circuit()
    for col in parsed_cols:
        basis_change = circuits.Circuit(
            cell.basis_change() for cell in col if cell is not None)
        body = circuits.Circuit(
            cell.operations() for cell in col if cell is not None)
        result += basis_change
        result += body
        result += basis_change**-1

    # Remap qubits if requested.
    if qubits is not None:
        qs = cast(Sequence['cirq.Qid'], qubits)

        def map_qubit(qubit: 'cirq.Qid') -> 'cirq.Qid':
            q = cast(devices.LineQubit, qubit)
            if q.x >= len(qs):
                raise IndexError(
                    f'Only {len(qs)} qubits specified, but the given quirk '
                    f'circuit used the qubit at offset {q.x}. Provide more '
                    f'qubits.')
            return qs[q.x]

        result = result.transform_qubits(map_qubit)

    return result


def _parse_col_cells(registry: Dict[str, CellMaker], col: int,
                     col_data: Any) -> List[Optional[Cell]]:
    if not isinstance(col_data, list):
        raise ValueError('col must be a list.\ncol: {!r}'.format(col_data))
    return [
        _parse_cell(registry, row, col, col_data[row])
        for row in range(len(col_data))
    ]


def _parse_cell(registry: Dict[str, CellMaker], row: int, col: int,
                entry: Any) -> Optional[Cell]:
    if entry == 1:
        return None

    key = None
    arg = None
    if isinstance(entry, dict):
        key = entry['id']
        arg = entry.get('arg', None)
    elif isinstance(entry, str):
        key = entry

    if isinstance(key, str) and key in registry:
        entry = registry[key]
        qubits = devices.LineQubit.range(row, row + entry.size)
        result = entry.maker(CellMakerArgs(qubits, arg, row=row, col=col))
        if isinstance(result, ops.Operation):
            return ExplicitOperationsCell([result])
        return result

    raise ValueError('Unrecognized column entry: {!r}'.format(entry))

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
    Iterable, Union, Mapping

import numpy as np

from cirq import devices, circuits, ops, protocols
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
        extra_cell_makers: Union[Dict[str, 'cirq.Gate'], Iterable[
            'cirq.contrib.quirk.cells.CellMaker']] = ()) -> 'cirq.Circuit':
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
        extra_cell_makers: Non-standard Quirk cells to accept. This can be
            used to parse URLs that come from a modified version of Quirk that
            includes gates that Quirk doesn't define. This can be specified
            as either a list of `cirq.contrib.quirk.cells.CellMaker` instances,
            or for more simple cases as a dictionary from a Quirk id string
            to a cirq Gate.

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
        ...     extra_cell_makers={'iswap': cirq.ISWAP}))
        0: ───iSwap───
              │
        1: ───iSwap───

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

    return quirk_json_to_circuit(data,
                                 qubits=qubits,
                                 extra_cell_makers=extra_cell_makers,
                                 quirk_url=quirk_url)


def quirk_json_to_circuit(
        data: dict,
        *,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        extra_cell_makers: Union[Dict[str, 'cirq.Gate'], Iterable[
            'cirq.contrib.quirk.cells.CellMaker']] = (),
        quirk_url: Optional[str] = None) -> 'cirq.Circuit':
    """Constructs a Cirq circuit from Quirk's JSON format.

    Args:
        data: Data parsed from quirk's JSON representation.
        qubits: Qubits to use in the circuit. See quirk_url_to_circuit.
        extra_cell_makers: Non-standard Quirk cells to accept. See
            quirk_url_to_circuit.
        quirk_url: If given, the original URL from which the JSON was parsed, as
            described in quirk_url_to_circuit.
    """

    def msg(error):
        if quirk_url is not None:
            return f'{error}\nURL={quirk_url}\nJSON={data}'
        else:
            return f'{error}\nJSON={data}'

    if not isinstance(data, dict):
        raise ValueError(msg('Circuit JSON must have a top-level dictionary.'))
    if not data.keys() <= {'cols', 'gates', 'init'}:
        raise ValueError(msg('Unrecognized Circuit JSON keys.'))
    if 'gates' in data:
        raise NotImplementedError(msg('Custom gates not supported yet.'))
    if 'cols' not in data:
        raise ValueError(msg('Circuit JSON dict must have a "cols" entry.'))

    cols = data['cols']
    if not isinstance(cols, list):
        raise ValueError(msg('Circuit JSON cols must be a list.'))

    # Collect registry of quirk cell types.
    if isinstance(extra_cell_makers, Mapping):
        extra_makers = [
            CellMaker(identifier=identifier,
                      size=protocols.num_qubits(gate),
                      maker=lambda args: gate(*args.qubits))
            for identifier, gate in extra_cell_makers.items()
        ]
    else:
        extra_makers = list(extra_cell_makers)
    registry = {
        entry.identifier: entry
        for entry in [*generate_all_quirk_cell_makers(), *extra_makers]
    }

    # Parse column json into cells.
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

    # Convert state initialization into operations.
    result.insert(0, _init_ops(data))

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


def _init_ops(data: Dict[str, Any]) -> 'cirq.OP_TREE':
    if 'init' not in data:
        return []
    init = data['init']
    if not isinstance(init, List):
        raise ValueError(f'Circuit JSON init must be a list but was {init!r}.')
    init_ops = []
    for i in range(len(init)):
        state = init[i]
        q = devices.LineQubit(i)
        if state == 0:
            pass
        elif state == 1:
            init_ops.append(ops.X(q))
        elif state == '+':
            init_ops.append(ops.Ry(np.pi / 2).on(q))
        elif state == '-':
            init_ops.append(ops.Ry(-np.pi / 2).on(q))
        elif state == 'i':
            init_ops.append(ops.Rx(-np.pi / 2).on(q))
        elif state == '-i':
            init_ops.append(ops.Rx(np.pi / 2).on(q))
        else:
            raise ValueError(f'Unrecognized init state: {state!r}')
    return ops.Moment(init_ops)


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

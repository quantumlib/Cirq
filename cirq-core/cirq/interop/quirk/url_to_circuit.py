# Copyright 2019 The Cirq Developers
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

from __future__ import annotations

import json
import urllib.parse
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np

from cirq import circuits, devices, ops, protocols
from cirq.interop.quirk.cells import (
    Cell,
    CellMaker,
    CellMakerArgs,
    CompositeCell,
    ExplicitOperationsCell,
    generate_all_quirk_cell_makers,
)
from cirq.interop.quirk.cells.parse import parse_matrix

if TYPE_CHECKING:
    import cirq


def quirk_url_to_circuit(
    quirk_url: str,
    *,
    qubits: Optional[Sequence[cirq.Qid]] = None,
    extra_cell_makers: Union[
        Dict[str, cirq.Gate], Iterable[cirq.interop.quirk.cells.CellMaker]
    ] = (),
    max_operation_count: int = 10**6,
) -> cirq.Circuit:
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
            as either a list of `cirq.interop.quirk.cells.CellMaker` instances,
            or for more simple cases as a dictionary from a Quirk id string
            to a cirq Gate.
        max_operation_count: If the number of operations in the circuit would
            exceed this value, the method raises a `ValueError` instead of
            attempting to construct the circuit. This is important to specify
            for servers parsing unknown input, because Quirk's format allows for
            a billion laughs attack in the form of nested custom gates.

    Examples:

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}'
    ... ))
    0: ───H───@───
              │
    1: ───────X───

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
    ...     qubits=[cirq.NamedQubit('Alice'), cirq.NamedQubit('Bob')]
    ... ))
    Alice: ───H───@───
                  │
    Bob: ─────────X───

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
    ...     extra_cell_makers={'iswap': cirq.ISWAP}))
    0: ───iSwap───
          │
    1: ───iSwap───

    >>> print(cirq.quirk_url_to_circuit(
    ...     'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
    ...     extra_cell_makers=[
    ...         cirq.interop.quirk.cells.CellMaker(
    ...             identifier='iswap',
    ...             size=2,
    ...             maker=lambda args: cirq.ISWAP(*args.qubits))
    ...     ]))
    0: ───iSwap───
          │
    1: ───iSwap───

    Returns:
        The parsed circuit.

    Raises:
        ValueError: Invalid circuit URL, or circuit would be larger than
            `max_operations_count`.
    """

    parsed_url = urllib.parse.urlparse(quirk_url)
    if not parsed_url.fragment:
        return circuits.Circuit()

    if not parsed_url.fragment.startswith('circuit='):
        raise ValueError(
            'Not a valid quirk url. The URL fragment (the part '
            'after the #) must start with "circuit=".\n'
            f'URL={quirk_url}'
        )

    # URL parser may not have fixed escaped characters in the fragment.
    json_text = parsed_url.fragment[len('circuit=') :]
    if '%22' in json_text:
        json_text = urllib.parse.unquote(json_text)

    data = json.loads(json_text)

    return quirk_json_to_circuit(
        data,
        qubits=qubits,
        extra_cell_makers=extra_cell_makers,
        quirk_url=quirk_url,
        max_operation_count=max_operation_count,
    )


def quirk_json_to_circuit(
    data: dict,
    *,
    qubits: Optional[Sequence[cirq.Qid]] = None,
    extra_cell_makers: Union[
        Dict[str, cirq.Gate], Iterable[cirq.interop.quirk.cells.CellMaker]
    ] = (),
    quirk_url: Optional[str] = None,
    max_operation_count: int = 10**6,
) -> cirq.Circuit:
    """Constructs a Cirq circuit from Quirk's JSON format.

    Args:
        data: Data parsed from quirk's JSON representation.
        qubits: Qubits to use in the circuit. See quirk_url_to_circuit.
        extra_cell_makers: Non-standard Quirk cells to accept. See
            quirk_url_to_circuit.
        quirk_url: If given, the original URL from which the JSON was parsed, as
            described in quirk_url_to_circuit.
        max_operation_count: If the number of operations in the circuit would
            exceed this value, the method raises a `ValueError` instead of
            attempting to construct the circuit. This is important to specify
            for servers parsing unknown input, because Quirk's format allows for
            a billion laughs attack in the form of nested custom gates.

    Examples:

    >>> print(cirq.quirk_json_to_circuit(
    ...     {"cols":[["H"], ["•", "X"]]}
    ... ))
    0: ───H───@───
              │
    1: ───────X───

    Returns:
        The parsed circuit.

    Raises:
        ValueError: Invalid circuit URL, or circuit would be larger than
            `max_operations_count`.
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

    # Collect registry of quirk cell types.
    if isinstance(extra_cell_makers, Mapping):
        extra_makers = [
            CellMaker(
                identifier=identifier,
                size=protocols.num_qubits(gate),
                maker=(lambda gate: lambda args: gate(*args.qubits))(gate),
            )
            for identifier, gate in extra_cell_makers.items()
        ]
    else:
        extra_makers = list(extra_cell_makers)
    registry = {
        entry.identifier: entry for entry in [*generate_all_quirk_cell_makers(), *extra_makers]
    }

    # Include custom gates in the registry.
    if 'gates' in data:
        if not isinstance(data['gates'], list):
            raise ValueError('"gates" JSON must be a list.')
        for custom_gate in data['gates']:
            _register_custom_gate(custom_gate, registry)

    # Parse out the circuit.
    comp = _parse_cols_into_composite_cell(data, registry)
    if max_operation_count is not None and comp.gate_count() > max_operation_count:
        raise ValueError(
            f'Quirk URL specifies a circuit with {comp.gate_count()} '
            f'operations, but max_operation_count={max_operation_count}.'
        )
    circuit = comp.circuit()

    # Convert state initialization into operations.
    circuit.insert(0, _init_ops(data))

    # Remap qubits if requested.
    if qubits is not None:
        qs = qubits

        def map_qubit(qubit: cirq.Qid) -> cirq.Qid:
            q = cast(devices.LineQubit, qubit)
            if q.x >= len(qs):
                raise IndexError(
                    f'Only {len(qs)} qubits specified, but the given quirk '
                    f'circuit used the qubit at offset {q.x}. Provide more '
                    f'qubits.'
                )
            return qs[q.x]

        circuit = circuit.transform_qubits(map_qubit)

    return circuit


def _parse_cols_into_composite_cell(
    data: Dict[str, Any], registry: Dict[str, CellMaker]
) -> CompositeCell:
    if not isinstance(data, Dict):
        raise ValueError('Circuit JSON must be a dictionary.')
    if 'cols' not in data:
        raise ValueError(f'Circuit JSON dict must have a "cols" entry.\nJSON={data}')
    cols = data['cols']
    if not isinstance(cols, list):
        raise ValueError(f'Circuit JSON cols must be a list.\nJSON={data}')

    # Parse column json into cells.
    parsed_cols: List[List[Optional[Cell]]] = []
    height = 0
    for i, col in enumerate(cols):
        parsed_col, h = _parse_col_cells_with_height(registry, i, col)
        height = max(height, h)
        parsed_cols.append(parsed_col)

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

    gate_count = sum(
        0 if cell is None else cell.gate_count() for col in parsed_cols for cell in col
    )

    return CompositeCell(height, parsed_cols, gate_count=gate_count)


def _register_custom_gate(gate_json: Any, registry: Dict[str, CellMaker]):
    if not isinstance(gate_json, Dict):
        raise ValueError(f'Custom gate json must be a dictionary.\nCustom gate json={gate_json!r}.')

    if 'id' not in gate_json:
        raise ValueError(f'Custom gate json must have an id key.\nCustom gate json={gate_json!r}.')
    identifier = gate_json['id']
    if identifier in registry:
        raise ValueError(f'Custom gate with duplicate identifier: {identifier!r}')

    if 'matrix' in gate_json and 'circuit' in gate_json:
        raise ValueError(
            f'Custom gate json cannot have both a matrix and a circuit.\n'
            f'Custom gate json={gate_json!r}.'
        )

    if 'matrix' in gate_json:
        if not isinstance(gate_json['matrix'], str):
            raise ValueError(
                f'Custom gate matrix json must be a string.\nCustom gate json={gate_json!r}.'
            )
        gate = ops.MatrixGate(parse_matrix(gate_json['matrix']))
        registry[identifier] = CellMaker(
            identifier=identifier,
            size=gate.num_qubits(),
            maker=lambda args: gate(*args.qubits[::-1]),
        )

    elif 'circuit' in gate_json:
        comp = _parse_cols_into_composite_cell(gate_json['circuit'], registry)
        registry[identifier] = CellMaker(
            identifier=identifier,
            size=comp.height,
            maker=lambda args: comp.with_line_qubits_mapped_to(list(args.qubits)),
        )

    else:
        raise ValueError(
            f'Custom gate json must have a matrix or a circuit.\n'
            f'Custom gate json={gate_json!r}.'
        )


def _init_ops(data: Dict[str, Any]) -> cirq.OP_TREE:
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
            init_ops.append(ops.ry(np.pi / 2).on(q))
        elif state == '-':
            init_ops.append(ops.ry(-np.pi / 2).on(q))
        elif state == 'i':
            init_ops.append(ops.rx(-np.pi / 2).on(q))
        elif state == '-i':
            init_ops.append(ops.rx(np.pi / 2).on(q))
        else:
            raise ValueError(f'Unrecognized init state: {state!r}')
    return circuits.Moment(init_ops)


def _parse_col_cells_with_height(
    registry: Dict[str, CellMaker], col: int, col_data: Any
) -> Tuple[List[Optional[Cell]], int]:
    if not isinstance(col_data, list):
        raise ValueError(f'col must be a list.\ncol: {col_data!r}')
    result = []
    height = 0
    for row in range(len(col_data)):
        cell, h = _parse_cell_with_height(registry, row, col, col_data[row])
        result.append(cell)
        height = max(height, h + row)
    return result, height


def _parse_cell_with_height(
    registry: Dict[str, CellMaker], row: int, col: int, entry: Any
) -> Tuple[Optional[Cell], int]:
    if entry == 1:
        return None, 0

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
            return ExplicitOperationsCell([result]), entry.size
        return result, entry.size

    raise ValueError(f'Unrecognized column entry: {entry!r}')

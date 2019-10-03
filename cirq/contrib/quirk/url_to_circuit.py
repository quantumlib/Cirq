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
from typing import Any, List, Dict, Optional

import cirq
from cirq.contrib.quirk.cells import (
    Cell,
    CellMaker,
    CellMakerArgs,
    generate_all_quirk_cell_makers,
)


def quirk_url_to_circuit(quirk_url: str) -> 'cirq.Circuit':

    parsed_url = urllib.parse.urlparse(quirk_url)
    if not parsed_url.fragment:
        return cirq.Circuit()

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
        entry.identifier: entry for entry in generate_all_quirk_cell_makers()
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

    # Extract circuit operations from modified cells.
    result = cirq.Circuit()
    for col in parsed_cols:
        basis_change = cirq.Circuit(
            cell.basis_change() for cell in col if cell is not None)
        body = cirq.Circuit(
            cell.operations() for cell in col if cell is not None)
        result += basis_change
        result += body
        result += basis_change**-1

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
        qubits = cirq.LineQubit.range(row, row + entry.size)
        return entry.maker(CellMakerArgs(qubits, arg, row=row, col=col))

    raise ValueError('Unrecognized column entry: {!r}'.format(entry))

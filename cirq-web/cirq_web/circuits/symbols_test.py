# Copyright 2021 The Cirq Developers
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
import pytest

import cirq
import cirq_web


class MockGateNoDiagramInfo(cirq.testing.SingleQubitGate):
    def __init__(self):
        super(MockGateNoDiagramInfo, self)


class MockGateUnimplementedDiagramInfo(cirq.testing.SingleQubitGate):
    def __init__(self):
        super(MockGateUnimplementedDiagramInfo, self)

    def _circuit_diagram_info_(self, args):
        return NotImplemented


def test_Operation3DSymbol_basic():
    wire_symbols = ['X']
    location_info = [{'row': 0, 'col': 0}]
    color_info = ['black']
    moment = 1

    symbol = cirq_web.circuits.symbols.Operation3DSymbol(
        wire_symbols, location_info, color_info, moment
    )

    actual = symbol.to_typescript()
    expected = {
        'wire_symbols': ['X'],
        'location_info': [{'row': 0, 'col': 0}],
        'color_info': ['black'],
        'moment': 1,
    }
    assert actual == expected


def test_resolve_operation_hadamard():
    mock_qubit = cirq.NamedQubit('mock')
    operation = cirq.H(mock_qubit)
    symbol_info = cirq_web.circuits.symbols.resolve_operation(
        operation, cirq_web.circuits.symbols.DEFAULT_SYMBOL_RESOLVERS
    )

    expected_labels = ['H']
    expected_colors = ['yellow']

    assert symbol_info.labels == expected_labels
    assert symbol_info.colors == expected_colors


def test_resolve_operation_x_pow():
    mock_qubit = cirq.NamedQubit('mock')
    operation = cirq.X(mock_qubit) ** 0.5
    symbol_info = cirq_web.circuits.symbols.resolve_operation(
        operation, cirq_web.circuits.symbols.DEFAULT_SYMBOL_RESOLVERS
    )

    expected_labels = ['X^0.5']
    expected_colors = ['black']

    assert symbol_info.labels == expected_labels
    assert symbol_info.colors == expected_colors


@pytest.mark.parametrize('custom_gate', [MockGateNoDiagramInfo, MockGateUnimplementedDiagramInfo])
def test_resolve_operation_invalid_diagram_info(custom_gate):
    mock_qubit = cirq.NamedQubit('mock')
    gate = custom_gate()
    operation = gate.on(mock_qubit)
    symbol_info = cirq_web.circuits.symbols.resolve_operation(
        operation, cirq_web.circuits.symbols.DEFAULT_SYMBOL_RESOLVERS
    )

    expected_labels = ['?']
    expected_colors = ['gray']

    assert symbol_info.labels == expected_labels
    assert symbol_info.colors == expected_colors


def test_unresolvable_operation_():
    mock_qubit = cirq.NamedQubit('mock')
    operation = cirq.X(mock_qubit)

    with pytest.raises(ValueError, match='Cannot resolve operation'):
        cirq_web.circuits.symbols.resolve_operation(operation, [])

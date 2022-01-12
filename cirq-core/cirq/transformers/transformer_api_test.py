# Copyright 2022 The Cirq Developers
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
from unittest import mock

import cirq
from cirq.circuits.circuit import CIRCUIT_TYPE

import pytest


@cirq.transformer
class MockTransformerClassCircuit:
    def __init__(self):
        self.mock = mock.Mock()

    def __call__(self, circuit: cirq.Circuit, context: cirq.TransformerContext) -> cirq.Circuit:
        self.mock(circuit, context)
        return circuit


@cirq.transformer
class MockTransformerClassFrozenCircuit:
    def __init__(self):
        self.mock = mock.Mock()

    def __call__(
        self, circuit: cirq.FrozenCircuit, context: cirq.TransformerContext
    ) -> cirq.FrozenCircuit:
        self.mock(circuit, context)
        return circuit


@cirq.transformer
class MockTransformerClassGeneric:
    def __init__(self):
        self.mock = mock.Mock()

    def __call__(self, circuit: CIRCUIT_TYPE, context: cirq.TransformerContext) -> CIRCUIT_TYPE:
        self.mock(circuit, context)
        return circuit


@cirq.transformer
def mock_transformer_method_circuit(
    circuit: cirq.Circuit, context: cirq.TransformerContext
) -> cirq.Circuit:
    if not hasattr(mock_transformer_method_circuit, 'mock'):
        mock_transformer_method_circuit.mock = mock.Mock()  # type: ignore
    mock_transformer_method_circuit.mock(circuit, context)  # type: ignore
    return circuit


@cirq.transformer
def mock_transformer_method_frozen_circuit(
    circuit: cirq.Circuit, context: cirq.TransformerContext
) -> cirq.Circuit:
    if not hasattr(mock_transformer_method_frozen_circuit, 'mock'):
        mock_transformer_method_frozen_circuit.mock = mock.Mock()  # type: ignore
    mock_transformer_method_frozen_circuit.mock(circuit, context)  # type: ignore
    return circuit


@cirq.transformer
def mock_transformer_method_generic(
    circuit: CIRCUIT_TYPE, context: cirq.TransformerContext
) -> CIRCUIT_TYPE:
    if not hasattr(mock_transformer_method_generic, 'mock'):
        mock_transformer_method_generic.mock = mock.Mock()  # type: ignore
    mock_transformer_method_generic.mock(circuit, context)  # type: ignore
    return circuit


@pytest.mark.parametrize(
    'context',
    [
        cirq.TransformerContext(),
        cirq.TransformerContext(logger=mock.Mock(), ignore_tags=('tag',)),
    ],
)
@pytest.mark.parametrize(
    'transformer',
    [
        MockTransformerClassCircuit(),
        MockTransformerClassFrozenCircuit(),
        MockTransformerClassGeneric(),
        mock_transformer_method_circuit,
        mock_transformer_method_frozen_circuit,
        mock_transformer_method_generic,
    ],
)
def test_transformer_decorator(context, transformer):
    circuit = cirq.Circuit(cirq.X(cirq.NamedQubit("a")))
    transformer(circuit, context)
    transformer.mock.assert_called_with(circuit, context)
    if context.logger is not None:
        transformer_name = (
            transformer.__name__ if hasattr(transformer, '__name__') else type(transformer).__name__
        )
        context.logger.register_initial.assert_called_with(circuit, transformer_name)
        context.logger.register_final.assert_called_with(circuit, transformer_name)

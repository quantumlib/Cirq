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
from cirq.transformers import LogLevel

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
    if not isinstance(context.logger, cirq.TransformerStatsLogger):
        transformer_name = (
            transformer.__name__ if hasattr(transformer, '__name__') else type(transformer).__name__
        )
        context.logger.register_initial.assert_called_with(circuit, transformer_name)
        context.logger.register_final.assert_called_with(circuit, transformer_name)


@cirq.transformer
class T1:
    def __call__(self, circuit: cirq.Circuit, context: cirq.TransformerContext) -> cirq.Circuit:
        context.logger.log("First Verbose Log", "of T1", level=LogLevel.DEBUG)
        context.logger.log("Second INFO Log", "of T1", level=LogLevel.INFO)
        context.logger.log("Third WARNING Log", "of T1", level=LogLevel.WARNING)
        return circuit + circuit[::-1]


t1 = T1()


@cirq.transformer
def t2(circuit: cirq.Circuit, context: cirq.TransformerContext) -> cirq.Circuit:
    context.logger.log("First INFO Log", "of T2 Start")
    circuit = t1(circuit, context)
    context.logger.log("Second INFO Log", "of T2 End")
    return circuit[::2]


@cirq.transformer
def t3(circuit: cirq.Circuit, context: cirq.TransformerContext) -> cirq.Circuit:
    context.logger.log("First INFO Log", "of T3 Start")
    circuit = t1(circuit, context)
    context.logger.log("Second INFO Log", "of T3 Middle")
    circuit = t2(circuit, context)
    context.logger.log("Third INFO Log", "of T3 End")
    return circuit


def test_transformer_stats_logger_raises():
    with pytest.raises(ValueError, match='No active transformer'):
        logger = cirq.TransformerStatsLogger()
        logger.log('test log')

    with pytest.raises(ValueError, match='No active transformer'):
        logger = cirq.TransformerStatsLogger()
        logger.register_initial(cirq.Circuit(), 'stage-1')
        logger.register_final(cirq.Circuit(), 'stage-1')
        logger.log('test log')

    with pytest.raises(ValueError, match='currently active transformer stage-2'):
        logger = cirq.TransformerStatsLogger()
        logger.register_initial(cirq.Circuit(), 'stage-2')
        logger.register_final(cirq.Circuit(), 'stage-3')


def test_transformer_stats_logger_show_levels(capfd):
    q = cirq.LineQubit.range(2)
    context = cirq.TransformerContext(logger=cirq.TransformerStatsLogger())
    initial_circuit = cirq.Circuit(cirq.H.on_each(*q), cirq.CNOT(*q))
    _ = t1(initial_circuit, context)
    info_line = 'LogLevel.INFO Second INFO Log of T1'
    debug_line = 'LogLevel.DEBUG First Verbose Log of T1'
    warning_line = 'LogLevel.WARNING Third WARNING Log of T1'

    for level in [LogLevel.ALL, LogLevel.DEBUG]:
        context.logger.show(level)
        out, _ = capfd.readouterr()
        assert all(line in out for line in [info_line, debug_line, warning_line])

    context.logger.show(LogLevel.INFO)
    out, _ = capfd.readouterr()
    assert info_line in out and warning_line in out and debug_line not in out

    context.logger.show(LogLevel.DEBUG)
    out, _ = capfd.readouterr()
    assert info_line in out and warning_line in out and debug_line in out

    context.logger.show(LogLevel.NONE)
    out, _ = capfd.readouterr()
    assert all(line not in out for line in [info_line, debug_line, warning_line])


def test_transformer_stats_logger_linear_and_nested(capfd):
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H.on_each(*q), cirq.CNOT(*q))
    context = cirq.TransformerContext(logger=cirq.TransformerStatsLogger())
    circuit = t1(circuit, context)
    circuit = t3(circuit, context)
    context.logger.show(LogLevel.ALL)
    out, _ = capfd.readouterr()
    assert (
        out.strip()
        == '''
Transformer-1: T1
Initial Circuit:
0: ───H───@───
          │
1: ───H───X───

 LogLevel.DEBUG First Verbose Log of T1
 LogLevel.INFO Second INFO Log of T1
 LogLevel.WARNING Third WARNING Log of T1

Final Circuit:
0: ───H───@───@───H───
          │   │
1: ───H───X───X───H───
----------------------------------------
Transformer-2: t3
Initial Circuit:
0: ───H───@───@───H───
          │   │
1: ───H───X───X───H───

 LogLevel.INFO First INFO Log of T3 Start
 LogLevel.INFO Second INFO Log of T3 Middle
 LogLevel.INFO Third INFO Log of T3 End

Final Circuit:
0: ───H───@───H───@───H───@───H───@───
          │       │       │       │
1: ───H───X───H───X───H───X───H───X───
----------------------------------------
    Transformer-3: T1
    Initial Circuit:
    0: ───H───@───@───H───
              │   │
    1: ───H───X───X───H───

     LogLevel.DEBUG First Verbose Log of T1
     LogLevel.INFO Second INFO Log of T1
     LogLevel.WARNING Third WARNING Log of T1

    Final Circuit:
    0: ───H───@───@───H───H───@───@───H───
              │   │           │   │
    1: ───H───X───X───H───H───X───X───H───
----------------------------------------
    Transformer-4: t2
    Initial Circuit:
    0: ───H───@───@───H───H───@───@───H───
              │   │           │   │
    1: ───H───X───X───H───H───X───X───H───

     LogLevel.INFO First INFO Log of T2 Start
     LogLevel.INFO Second INFO Log of T2 End

    Final Circuit:
    0: ───H───@───H───@───H───@───H───@───
              │       │       │       │
    1: ───H───X───H───X───H───X───H───X───
----------------------------------------
        Transformer-5: T1
        Initial Circuit:
        0: ───H───@───@───H───H───@───@───H───
                  │   │           │   │
        1: ───H───X───X───H───H───X───X───H───

         LogLevel.DEBUG First Verbose Log of T1
         LogLevel.INFO Second INFO Log of T1
         LogLevel.WARNING Third WARNING Log of T1

        Final Circuit:
        0: ───H───@───@───H───H───@───@───H───H───@───@───H───H───@───@───H───
                  │   │           │   │           │   │           │   │
        1: ───H───X───X───H───H───X───X───H───H───X───X───H───H───X───X───H───
----------------------------------------
'''.strip()
    )

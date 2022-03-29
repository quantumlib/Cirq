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
from typing import Optional

import cirq
from cirq.transformers.transformer_api import LogLevel

import pytest


@cirq.transformer()
class MockTransformerClass:
    def __init__(self):
        self.mock = mock.Mock()

    def __call__(
        self, circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext] = None
    ) -> cirq.Circuit:
        self.mock(circuit, context)
        return circuit.unfreeze()


@cirq.value_equality
class CustomArg:
    def __init__(self, x: int = 0):
        self._x = x

    def _value_equality_values_(self):
        return (self._x,)


@cirq.transformer
class MockTransformerClassWithDefaults:
    def __init__(self):
        self.mock = mock.Mock()

    def __call__(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        context: Optional[cirq.TransformerContext] = cirq.TransformerContext(),
        atol: float = 1e-4,
        custom_arg: CustomArg = CustomArg(),
    ) -> cirq.AbstractCircuit:
        self.mock(circuit, context, atol, custom_arg)
        return circuit[::-1]


@cirq.transformer(add_deep_support=True)
class MockTransformerClassSupportsDeep(MockTransformerClass):
    pass


def make_transformer_func_with_defaults() -> cirq.TRANSFORMER:
    my_mock = mock.Mock()

    @cirq.transformer
    def func(
        circuit: cirq.AbstractCircuit,
        *,
        context: Optional[cirq.TransformerContext] = cirq.TransformerContext(),
        atol: float = 1e-4,
        custom_arg: CustomArg = CustomArg(),
    ) -> cirq.FrozenCircuit:
        my_mock(circuit, context, atol, custom_arg)
        return circuit.freeze()

    func.mock = my_mock  # type: ignore
    return func


def make_transformer_func(add_deep_support: bool = False) -> cirq.TRANSFORMER:
    my_mock = mock.Mock()

    @cirq.transformer(add_deep_support=add_deep_support)
    def mock_tranformer_func(
        circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext] = None
    ) -> cirq.Circuit:
        my_mock(circuit, context)
        return circuit.unfreeze()

    mock_tranformer_func.mock = my_mock  # type: ignore
    return mock_tranformer_func


@pytest.mark.parametrize(
    'context',
    [
        cirq.TransformerContext(),
        cirq.TransformerContext(logger=mock.Mock(), tags_to_ignore=('tag',)),
    ],
)
@pytest.mark.parametrize(
    'transformer',
    [MockTransformerClass(), make_transformer_func()],
)
def test_transformer_decorator(context, transformer):
    circuit = cirq.Circuit(cirq.X(cirq.NamedQubit("a")))
    transformer(circuit, context=context)
    transformer.mock.assert_called_with(circuit, context)
    if not isinstance(context.logger, cirq.TransformerLogger):
        transformer_name = (
            transformer.__name__ if hasattr(transformer, '__name__') else type(transformer).__name__
        )
        context.logger.register_initial.assert_called_with(circuit, transformer_name)
        context.logger.register_final.assert_called_with(circuit, transformer_name)


@pytest.mark.parametrize(
    'transformer',
    [
        MockTransformerClassWithDefaults(),
        make_transformer_func_with_defaults(),
    ],
)
def test_transformer_decorator_with_defaults(transformer):
    circuit = cirq.Circuit(cirq.X(cirq.NamedQubit("a")))
    context = cirq.TransformerContext(tags_to_ignore=("tags", "to", "ignore"))
    transformer(circuit)
    transformer.mock.assert_called_with(circuit, cirq.TransformerContext(), 1e-4, CustomArg())
    transformer(circuit, context=context, atol=1e-3)
    transformer.mock.assert_called_with(circuit, context, 1e-3, CustomArg())
    transformer(circuit, context=context, custom_arg=CustomArg(10))
    transformer.mock.assert_called_with(circuit, context, 1e-4, CustomArg(10))
    transformer(circuit, context=context, atol=1e-2, custom_arg=CustomArg(12))
    transformer.mock.assert_called_with(circuit, context, 1e-2, CustomArg(12))


@pytest.mark.parametrize(
    'transformer, supports_deep',
    [
        (MockTransformerClass(), False),
        (make_transformer_func(), False),
        (MockTransformerClassSupportsDeep(), True),
        (make_transformer_func(add_deep_support=True), True),
    ],
)
def test_transformer_decorator_adds_support_for_deep(transformer, supports_deep):
    q = cirq.NamedQubit("q")
    c_nested_x = cirq.FrozenCircuit(cirq.X(q))
    c_nested_y = cirq.FrozenCircuit(cirq.Y(q))
    c_nested_xy = cirq.FrozenCircuit(
        cirq.CircuitOperation(c_nested_x).repeat(5).with_tags("ignore"),
        cirq.CircuitOperation(c_nested_y).repeat(7).with_tags("preserve_tag"),
    )
    c_nested_yx = cirq.FrozenCircuit(
        cirq.CircuitOperation(c_nested_y).repeat(7).with_tags("ignore"),
        cirq.CircuitOperation(c_nested_x).repeat(5).with_tags("preserve_tag"),
    )
    c_orig = cirq.Circuit(
        cirq.CircuitOperation(c_nested_xy).repeat(4),
        cirq.CircuitOperation(c_nested_x).repeat(5).with_tags("ignore"),
        cirq.CircuitOperation(c_nested_y).repeat(6),
        cirq.CircuitOperation(c_nested_yx).repeat(7),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    transformer(c_orig, context=context)
    expected_calls = [mock.call(c_orig, context)]
    if supports_deep:
        expected_calls = [
            mock.call(c_nested_y, context),  # c_orig --> xy --> y
            mock.call(c_nested_xy, context),  # c_orig --> xy
            mock.call(c_nested_y, context),  # c_orig --> y
            mock.call(c_nested_x, context),  # c_orig --> yx --> x
            mock.call(c_nested_yx, context),  # c_orig --> yx
            mock.call(c_orig, context),  # c_orig
        ]
    transformer.mock.assert_has_calls(expected_calls)


@cirq.transformer
class T1:
    def __call__(
        self, circuit: cirq.AbstractCircuit, context: Optional[cirq.TransformerContext] = None
    ) -> cirq.AbstractCircuit:
        assert context is not None
        context.logger.log("First Verbose Log", "of T1", level=LogLevel.DEBUG)
        context.logger.log("Second INFO Log", "of T1", level=LogLevel.INFO)
        context.logger.log("Third WARNING Log", "of T1", level=LogLevel.WARNING)
        return cirq.Circuit(*circuit, *circuit[::-1])


t1 = T1()


@cirq.transformer
def t2(
    circuit: cirq.AbstractCircuit, context: Optional[cirq.TransformerContext] = None
) -> cirq.FrozenCircuit:
    assert context is not None
    context.logger.log("First INFO Log", "of T2 Start")
    circuit = t1(circuit, context=context)
    context.logger.log("Second INFO Log", "of T2 End")
    return circuit[::2].freeze()


@cirq.transformer
def t3(
    circuit: cirq.AbstractCircuit, context: Optional[cirq.TransformerContext] = None
) -> cirq.Circuit:
    assert context is not None
    context.logger.log("First INFO Log", "of T3 Start")
    circuit = t1(circuit, context=context)
    context.logger.log("Second INFO Log", "of T3 Middle")
    circuit = t2(circuit, context=context)
    context.logger.log("Third INFO Log", "of T3 End")
    return circuit.unfreeze()


def test_transformer_stats_logger_raises():
    with pytest.raises(ValueError, match='No active transformer'):
        logger = cirq.TransformerLogger()
        logger.log('test log')

    with pytest.raises(ValueError, match='No active transformer'):
        logger = cirq.TransformerLogger()
        logger.register_initial(cirq.Circuit(), 'stage-1')
        logger.register_final(cirq.Circuit(), 'stage-1')
        logger.log('test log')

    with pytest.raises(ValueError, match='currently active transformer stage-2'):
        logger = cirq.TransformerLogger()
        logger.register_initial(cirq.Circuit(), 'stage-2')
        logger.register_final(cirq.Circuit(), 'stage-3')


def test_transformer_stats_logger_show_levels(capfd):
    q = cirq.LineQubit.range(2)
    context = cirq.TransformerContext(logger=cirq.TransformerLogger())
    initial_circuit = cirq.Circuit(cirq.H.on_each(*q), cirq.CNOT(*q))
    _ = t1(initial_circuit, context=context)
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
    context = cirq.TransformerContext(logger=cirq.TransformerLogger())
    circuit = t1(circuit, context=context)
    circuit = t3(circuit, context=context)
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

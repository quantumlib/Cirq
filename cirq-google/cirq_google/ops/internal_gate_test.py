# Copyright 2023 The Cirq Developers
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

import numpy as np
import pytest

import cirq
import cirq_google
from cirq_google.ops import internal_gate
from cirq_google.serialization import arg_func_langs


def test_internal_gate():
    g = cirq_google.InternalGate(
        gate_name="CouplerDelayZ",
        gate_module='internal_module',
        num_qubits=2,
        delay=1,
        zpa=0.0,
        zpl=None,
    )
    assert str(g) == 'internal_module.CouplerDelayZ(delay=1, zpa=0.0, zpl=None)'
    want_repr = (
        "cirq_google.InternalGate(gate_name='CouplerDelayZ', "
        "gate_module='internal_module', num_qubits=2, "
        "delay=1, zpa=0.0, zpl=None)"
    )
    assert repr(g) == want_repr
    assert cirq.qid_shape(g) == (2, 2)


def test_internal_gate_with_no_args():
    g = cirq_google.InternalGate(gate_name="GateWithNoArgs", gate_module='test', num_qubits=3)
    assert str(g) == 'test.GateWithNoArgs()'
    want_repr = (
        "cirq_google.InternalGate(gate_name='GateWithNoArgs', gate_module='test', num_qubits=3)"
    )
    assert repr(g) == want_repr
    assert cirq.qid_shape(g) == (2, 2, 2)


def test_internal_gate_with_hashable_args_is_hashable():
    hashable = cirq_google.InternalGate(
        gate_name="GateWithHashableArgs",
        gate_module='test',
        num_qubits=3,
        foo=1,
        bar="2",
        baz=(("a", 1),),
    )
    _ = hash(hashable)

    unhashable = cirq_google.InternalGate(
        gate_name="GateWithHashableArgs",
        gate_module='test',
        num_qubits=3,
        foo=1,
        bar="2",
        baz={"a": 1},
    )
    with pytest.raises(TypeError, match="unhashable"):
        _ = hash(unhashable)


def test_internal_gate_with_custom_function_repr():
    x = np.linspace(-1, 1, 10)
    y = x**2
    encoded_func = internal_gate.function_points_to_proto(x=x, y=y)

    gate = internal_gate.InternalGate(
        gate_name='GateWithFunction',
        gate_module='test',
        num_qubits=2,
        custom_args={'func': encoded_func},
    )

    assert repr(gate) == (
        "cirq_google.InternalGate(gate_name='GateWithFunction', "
        f"gate_module='test', num_qubits=2, custom_args={gate.custom_args})"
    )

    assert str(gate) == (f"test.GateWithFunction(func={encoded_func})")

    with pytest.raises(ValueError):
        _ = cirq.to_json(gate)


def test_internal_gate_with_custom_function_round_trip():
    original_func = lambda x: x**2
    x = np.linspace(-1, 1, 10)
    y = original_func(x)
    encoded_func = internal_gate.function_points_to_proto(x=x, y=y)

    gate = internal_gate.InternalGate(
        gate_name='GateWithFunction',
        gate_module='test',
        num_qubits=2,
        custom_args={'func': encoded_func},
    )

    msg = arg_func_langs.internal_gate_arg_to_proto(gate)

    new_gate = arg_func_langs.internal_gate_from_proto(msg)

    func_proto = new_gate.custom_args['func'].function_interpolation_data

    np.testing.assert_allclose(x, func_proto.x_values)
    np.testing.assert_allclose(y, func_proto.y_values)


def test_function_points_to_proto_invalid_args_raise():
    x = np.linspace(-1, 1, 10)
    y = x + 1

    with pytest.raises(ValueError, match='The free variable must be one dimensional'):
        _ = internal_gate.function_points_to_proto(np.zeros((10, 2)), y)

    with pytest.raises(ValueError, match='sorted in increasing order'):
        _ = internal_gate.function_points_to_proto(x[::-1], y)

    with pytest.raises(ValueError, match='Mismatch between number of points in x and y'):
        _ = internal_gate.function_points_to_proto(x, np.linspace(-1, 1, 40))

    with pytest.raises(ValueError, match='The dependent variable must be one dimensional'):
        _ = internal_gate.function_points_to_proto(x, np.zeros((10, 2)))


def test_custom_gates_are_taken_into_equality():
    msg1 = internal_gate.function_points_to_proto(x=np.linspace(0, 1, 10), y=np.random.random(10))
    msg2 = internal_gate.function_points_to_proto(x=np.linspace(-1, 0, 10), y=np.random.random(10))
    g1 = internal_gate.InternalGate('test', 'test', custom_args={'f1': msg1})
    g2 = internal_gate.InternalGate('test', 'test', custom_args={'f1': msg1})
    g3 = internal_gate.InternalGate('test', 'test', custom_args={'f1': msg2})

    assert g1 == g2
    assert g1 != g3

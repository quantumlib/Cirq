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

import cirq
import cirq_google
import pytest


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

# Copyright 2018 The Cirq Developers
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


class NoMethod:
    pass


class ReturnsNotImplemented:
    def controlled_by(self, *control_qubits):
        return NotImplemented

p = cirq.NamedQubit('p')
q = cirq.NamedQubit('q')
dummy_gate = cirq.SingleQubitGate()
dummy_op = cirq.GateOperation(cirq.X, [p])

@pytest.mark.parametrize('controllee', (
    NoMethod(),
    object(),
    ReturnsNotImplemented(),
))
def test_controlless(controllee):
    assert cirq.protocols.control(controllee, None, None) is None
    assert cirq.protocols.control(controllee, [p],
                                  NotImplemented) is NotImplemented
    assert cirq.protocols.control(controllee, [p], None) is None
    assert cirq.protocols.control(controllee, [p,q], None) is None


def test_controlled_by_error():
    with pytest.raises(TypeError, match="returned NotImplemented"):
        _ = cirq.protocols.control(ReturnsNotImplemented(), [p])
    with pytest.raises(TypeError, match="no controlled_by method"):
        _ = cirq.protocols.control(NoMethod(), [p])


@pytest.mark.parametrize('controllee,control_qubits,out', (
    (dummy_gate, [p], cirq.ControlledGate(dummy_gate, [p])),
    (dummy_op, [q], cirq.ControlledOperation([q], dummy_op)),
))
def test_pow_with_result(controllee, control_qubits, out):
    assert (cirq.protocols.control(controllee, control_qubits) ==
            cirq.protocols.control(controllee, control_qubits, default=None) ==
            out)


def test_op_tree_control():
    gs = [cirq.SingleQubitGate() for _ in range(10)]
    op_tree = [
        cirq.GateOperation(gs[i], [cirq.NamedQubit(str(i))])
        for i in range(10)
    ]
    controls = cirq.LineQubit.range(2)
    controlled_op_tree = cirq.protocols.control(op_tree, controls)
    expected= [
        cirq.ControlledOperation(
            controls, cirq.GateOperation(gs[i], [cirq.NamedQubit(str(i))]))
        for i in range(10)
    ]
    assert cirq.freeze_op_tree(controlled_op_tree) == tuple(expected)
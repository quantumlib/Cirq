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

import numpy as np

import cirq


class No:
    pass


class No1:

    def _has_stabilizer_effect_(self):
        return NotImplemented


class No2:

    def _has_stabilizer_effect_(self):
        return None


class No3:

    def _has_stabilizer_effect_(self):
        return False


class Yes:

    def _has_stabilizer_effect_(self):
        return True


class EmptyOp(cirq.Operation):
    """A trivial operation."""

    @property
    def qubits(self):
        # coverage: ignore
        return ()

    def with_qubits(self, *new_qubits):
        # coverage: ignore
        return self


class NoOp(EmptyOp):

    @property
    def gate(self):
        return No()


class NoOp1(EmptyOp):

    @property
    def gate(self):
        return No1()


class NoOp2(EmptyOp):

    @property
    def gate(self):
        return No2()


class NoOp3(EmptyOp):

    @property
    def gate(self):
        return No3()


class YesOp(EmptyOp):

    @property
    def gate(self):
        return Yes()


class OpWithUnitary(EmptyOp):

    def __init__(self, unitary):
        self.unitary = unitary

    def _unitary_(self):
        return self.unitary


def test_inconclusive():
    assert not cirq.has_stabilizer_effect(object())
    assert not cirq.has_stabilizer_effect('boo')
    assert not cirq.has_stabilizer_effect(cirq.SingleQubitGate())
    assert not cirq.has_stabilizer_effect(No())
    assert not cirq.has_stabilizer_effect(NoOp())


def test_via_has_stabilizer_effect_method():
    assert not cirq.has_stabilizer_effect(No1())
    assert not cirq.has_stabilizer_effect(No2())
    assert not cirq.has_stabilizer_effect(No3())
    assert cirq.has_stabilizer_effect(Yes())


def test_via_gate_of_op():
    assert cirq.has_stabilizer_effect(YesOp())
    assert not cirq.has_stabilizer_effect(NoOp1())
    assert not cirq.has_stabilizer_effect(NoOp2())
    assert not cirq.has_stabilizer_effect(NoOp3())


def test_via_unitary():
    op1 = OpWithUnitary(np.array([[0, 1], [1, 0]]))
    assert cirq.has_stabilizer_effect(op1)

    op2 = OpWithUnitary(np.array([[0, 1j], [1j, 0]]))
    assert cirq.has_stabilizer_effect(op2)

    op3 = OpWithUnitary(np.array([[1, 0], [0, np.sqrt(1j)]]))
    assert not cirq.has_stabilizer_effect(op3)


def test_via_unitary_not_supported():
    # Unitaries larger than 2x2 are not yet supported.
    op = OpWithUnitary(cirq.unitary(cirq.CNOT))
    assert not cirq.has_stabilizer_effect(op)
    assert not cirq.has_stabilizer_effect(op)
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
from typing import Optional

import numpy as np
import pytest

import cirq
from cirq import testing

m0: np.ndarray = np.array([])
# yapf: disable
# X on one qubit
m1 = np.array([[0, 1],
               [1, 0]])
# X on two qubits
m2 = np.array([[0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [1, 0, 0, 0]])
# X on qubit 1 and 3
m3 = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0]])
# yapf: enable
a = cirq.NamedQubit('a')
b = cirq.NamedQubit('b')
c = cirq.NamedQubit('c')


class NoMethod:
    pass


class ReturnsNotImplemented(cirq.Gate):
    def _has_unitary_(self):
        return NotImplemented

    def _unitary_(self):
        return NotImplemented

    def num_qubits(self):
        return 1


class ReturnsMatrix(cirq.Gate):
    def _unitary_(self) -> np.ndarray:
        return m1

    def num_qubits(self):
        return 1  # coverage: ignore


class FullyImplemented(cirq.Gate):
    def __init__(self, unitary_value):
        self.unitary_value = unitary_value

    def _has_unitary_(self) -> bool:
        return self.unitary_value

    def _unitary_(self) -> Optional[np.ndarray]:
        if not self.unitary_value:
            return None
        return m1

    def num_qubits(self):
        return 1


class DecomposableGate(cirq.Gate):
    def __init__(self, unitary_value):
        self.unitary_value = unitary_value

    def num_qubits(self):
        return 1

    def _decompose_(self, qubits):
        yield FullyImplemented(self.unitary_value).on(qubits[0])


class DecomposableOperation(cirq.Operation):
    qubits = ()
    with_qubits = NotImplemented

    def __init__(self, qubits, unitary_value):
        self.qubits = qubits
        self.unitary_value = unitary_value

    def _decompose_(self):
        for q in self.qubits:
            yield FullyImplemented(self.unitary_value)(q)


class DecomposableOrder(cirq.Operation):
    qubits = ()
    with_qubits = NotImplemented

    def __init__(self, qubits):
        self.qubits = qubits

    def _decompose_(self):
        yield FullyImplemented(True)(self.qubits[2])
        yield FullyImplemented(True)(self.qubits[0])


class DecomposableNoUnitary(cirq.Operation):
    qubits = ()
    with_qubits = NotImplemented

    def __init__(self, qubits):
        self.qubits = qubits

    def _decompose_(self):
        for q in self.qubits:
            yield ReturnsNotImplemented()(q)


class DummyOperation(cirq.Operation):
    qubits = ()
    with_qubits = NotImplemented

    def __init__(self, qubits):
        self.qubits = qubits

    def _decompose_(self):
        return ()


class DummyComposite:
    def _decompose_(self):
        return ()


class OtherComposite:
    def _decompose_(self):
        yield cirq.X(cirq.LineQubit(0))
        yield cirq.X(cirq.LineQubit(3))


def test_unitary():
    with pytest.raises(TypeError, match='unitary effect'):
        _ = cirq.unitary(NoMethod())
    with pytest.raises(TypeError, match='unitary effect'):
        _ = cirq.unitary(ReturnsNotImplemented())
    assert cirq.unitary(ReturnsMatrix()) is m1

    assert cirq.unitary(NoMethod(), None) is None
    assert cirq.unitary(ReturnsNotImplemented(), None) is None
    assert cirq.unitary(ReturnsMatrix(), None) is m1

    assert cirq.unitary(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.unitary(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert cirq.unitary(ReturnsMatrix(), NotImplemented) is m1

    assert cirq.unitary(NoMethod(), 1) == 1
    assert cirq.unitary(ReturnsNotImplemented(), 1) == 1
    assert cirq.unitary(ReturnsMatrix(), 1) is m1

    assert cirq.unitary(NoMethod(), m0) is m0
    assert cirq.unitary(ReturnsNotImplemented(), m0) is m0
    assert cirq.unitary(ReturnsMatrix(), m0) is m1
    assert cirq.unitary(FullyImplemented(True), m0) is m1
    assert cirq.unitary(FullyImplemented(False), default=None) is None


def test_has_unitary():
    assert not cirq.has_unitary(NoMethod())
    assert not cirq.has_unitary(ReturnsNotImplemented())
    assert cirq.has_unitary(ReturnsMatrix())

    # Explicit function should override
    assert cirq.has_unitary(FullyImplemented(True))
    assert not cirq.has_unitary(FullyImplemented(False))


def _test_gate_that_allocates_qubits(gate):
    from cirq.protocols.unitary_protocol import _strat_unitary_from_decompose

    op = gate.on(*cirq.LineQubit.range(cirq.num_qubits(gate)))
    moment = cirq.Moment(op)
    circuit = cirq.FrozenCircuit(op)
    circuit_op = cirq.CircuitOperation(circuit)
    for val in [gate, op, moment, circuit, circuit_op]:
        unitary_from_strat = _strat_unitary_from_decompose(val)
        assert unitary_from_strat is not None
        np.testing.assert_allclose(unitary_from_strat, gate.narrow_unitary())


@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 10))
@pytest.mark.parametrize('phase_state', [0, 1])
@pytest.mark.parametrize('target_bitsize', [1, 2, 3])
@pytest.mark.parametrize('ancilla_bitsize', [1, 4])
def test_decompose_gate_that_allocates_clean_qubits(
    theta: float, phase_state: int, target_bitsize: int, ancilla_bitsize: int
):

    gate = testing.PhaseUsingCleanAncilla(theta, phase_state, target_bitsize, ancilla_bitsize)
    _test_gate_that_allocates_qubits(gate)


@pytest.mark.parametrize('phase_state', [0, 1])
@pytest.mark.parametrize('target_bitsize', [1, 2, 3])
@pytest.mark.parametrize('ancilla_bitsize', [1, 4])
def test_decompose_gate_that_allocates_dirty_qubits(
    phase_state: int, target_bitsize: int, ancilla_bitsize: int
):

    gate = testing.PhaseUsingDirtyAncilla(phase_state, target_bitsize, ancilla_bitsize)
    _test_gate_that_allocates_qubits(gate)


def test_decompose_and_get_unitary():
    from cirq.protocols.unitary_protocol import _strat_unitary_from_decompose

    np.testing.assert_allclose(_strat_unitary_from_decompose(DecomposableOperation((a,), True)), m1)
    np.testing.assert_allclose(
        _strat_unitary_from_decompose(DecomposableOperation((a, b), True)), m2
    )
    np.testing.assert_allclose(_strat_unitary_from_decompose(DecomposableOrder((a, b, c))), m3)
    np.testing.assert_allclose(_strat_unitary_from_decompose(DummyOperation((a,))), np.eye(2))
    np.testing.assert_allclose(_strat_unitary_from_decompose(DummyOperation((a, b))), np.eye(4))
    np.testing.assert_allclose(_strat_unitary_from_decompose(DummyComposite()), np.eye(1))
    np.testing.assert_allclose(_strat_unitary_from_decompose(OtherComposite()), m2)


def test_decomposed_has_unitary():
    # Gates
    assert cirq.has_unitary(DecomposableGate(True))
    assert not cirq.has_unitary(DecomposableGate(False))

    # GateOperations
    assert cirq.has_unitary(DecomposableGate(True).on(a))
    assert not cirq.has_unitary(DecomposableGate(False).on(a))

    # Operations
    assert cirq.has_unitary(DecomposableOperation((a, b), True))
    assert cirq.has_unitary(DummyOperation((a,)))
    assert cirq.has_unitary(DummyOperation((a, b)))

    # No qid shape
    assert cirq.has_unitary(DummyComposite())
    assert cirq.has_unitary(OtherComposite())


def test_decomposed_unitary():
    # Gates
    np.testing.assert_allclose(cirq.unitary(DecomposableGate(True)), m1)

    # GateOperations
    np.testing.assert_allclose(cirq.unitary(DecomposableGate(True).on(a)), m1)

    # Operations
    np.testing.assert_allclose(cirq.unitary(DecomposableOperation((a,), True)), m1)
    np.testing.assert_allclose(cirq.unitary(DecomposableOperation((a, b), True)), m2)
    np.testing.assert_allclose(cirq.unitary(DecomposableOrder((a, b, c))), m3)
    np.testing.assert_allclose(cirq.unitary(DummyOperation((a,))), np.eye(2))
    np.testing.assert_allclose(cirq.unitary(DummyOperation((a, b))), np.eye(4))
    assert cirq.unitary(DecomposableNoUnitary((a,)), None) is None

    # No qid shape
    np.testing.assert_allclose(cirq.unitary(DummyComposite()), np.eye(1))
    np.testing.assert_allclose(cirq.unitary(OtherComposite()), m2)


def test_unitary_from_apply_unitary():
    class ApplyGate(cirq.Gate):
        def num_qubits(self):
            return 1

        def _apply_unitary_(self, args):
            return cirq.apply_unitary(cirq.X(cirq.LineQubit(0)), args)

    class UnknownType:
        def _apply_unitary_(self, args):
            assert False

    class ApplyGateNotUnitary(cirq.Gate):
        def num_qubits(self):
            return 1

        def _apply_unitary_(self, args):
            return None

    class ApplyOp(cirq.Operation):
        def __init__(self, q):
            self.q = q

        @property
        def qubits(self):
            return (self.q,)

        def with_qubits(self, *new_qubits):
            # coverage: ignore
            return ApplyOp(*new_qubits)

        def _apply_unitary_(self, args):
            return cirq.apply_unitary(cirq.X(self.q), args)

    assert cirq.has_unitary(ApplyGate())
    assert cirq.has_unitary(ApplyOp(cirq.LineQubit(0)))
    assert not cirq.has_unitary(ApplyGateNotUnitary())
    assert not cirq.has_unitary(UnknownType())

    np.testing.assert_allclose(cirq.unitary(ApplyGate()), np.array([[0, 1], [1, 0]]))
    np.testing.assert_allclose(cirq.unitary(ApplyOp(cirq.LineQubit(0))), np.array([[0, 1], [1, 0]]))
    assert cirq.unitary(ApplyGateNotUnitary(), default=None) is None
    assert cirq.unitary(UnknownType(), default=None) is None

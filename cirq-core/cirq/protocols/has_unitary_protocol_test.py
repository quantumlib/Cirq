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
import pytest

import cirq


def test_inconclusive():
    class No:
        pass

    assert not cirq.has_unitary(object())
    assert not cirq.has_unitary('boo')
    assert not cirq.has_unitary(No())


@pytest.mark.parametrize(
    'measurement_gate', (cirq.MeasurementGate(1, 'a'), cirq.PauliMeasurementGate([cirq.X], 'a'))
)
def test_fail_fast_measure(measurement_gate):
    assert not cirq.has_unitary(measurement_gate)

    qubit = cirq.NamedQubit('q0')
    circuit = cirq.Circuit()
    circuit += measurement_gate(qubit)
    circuit += cirq.H(qubit)
    assert not cirq.has_unitary(circuit)


def test_fail_fast_measure_large_memory():
    num_qubits = 100
    measurement_op = cirq.MeasurementGate(num_qubits, 'a').on(*cirq.LineQubit.range(num_qubits))
    assert not cirq.has_unitary(measurement_op)


def test_via_unitary():
    class No1:
        def _unitary_(self):
            return NotImplemented

    class No2:
        def _unitary_(self):
            return None

    class Yes:
        def _unitary_(self):
            return np.array([[1]])

    assert not cirq.has_unitary(No1())
    assert not cirq.has_unitary(No2())
    assert cirq.has_unitary(Yes())
    assert cirq.has_unitary(Yes(), allow_decompose=False)


def test_via_apply_unitary():
    class No1(EmptyOp):
        def _apply_unitary_(self, args):
            return None

    class No2(EmptyOp):
        def _apply_unitary_(self, args):
            return NotImplemented

    class No3(cirq.testing.SingleQubitGate):
        def _apply_unitary_(self, args):
            return NotImplemented

    class No4:  # A non-operation non-gate.
        def _apply_unitary_(self, args):
            assert False  # Because has_unitary doesn't understand how to call.

    class Yes1(EmptyOp):
        def _apply_unitary_(self, args):
            return args.target_tensor

    class Yes2(cirq.testing.SingleQubitGate):
        def _apply_unitary_(self, args):
            return args.target_tensor

    assert cirq.has_unitary(Yes1())
    assert cirq.has_unitary(Yes1(), allow_decompose=False)
    assert cirq.has_unitary(Yes2())
    assert not cirq.has_unitary(No1())
    assert not cirq.has_unitary(No2())
    assert not cirq.has_unitary(No3())
    assert not cirq.has_unitary(No4())


def test_via_decompose():
    class Yes1:
        def _decompose_(self):
            return []

    class Yes2:
        def _decompose_(self):
            return [cirq.X(cirq.LineQubit(0))]

    class No1:
        def _decompose_(self):
            return [cirq.depolarize(0.5).on(cirq.LineQubit(0))]

    class No2:
        def _decompose_(self):
            return None

    class No3:
        def _decompose_(self):
            return NotImplemented

    assert cirq.has_unitary(Yes1())
    assert cirq.has_unitary(Yes2())
    assert not cirq.has_unitary(No1())
    assert not cirq.has_unitary(No2())
    assert not cirq.has_unitary(No3())

    assert not cirq.has_unitary(Yes1(), allow_decompose=False)
    assert not cirq.has_unitary(Yes2(), allow_decompose=False)
    assert not cirq.has_unitary(No1(), allow_decompose=False)


def test_via_has_unitary():
    class No1:
        def _has_unitary_(self):
            return NotImplemented

    class No2:
        def _has_unitary_(self):
            return False

    class Yes:
        def _has_unitary_(self):
            return True

    assert not cirq.has_unitary(No1())
    assert not cirq.has_unitary(No2())
    assert cirq.has_unitary(Yes())


def test_order():
    class Yes1(EmptyOp):
        def _has_unitary_(self):
            return True

        def _decompose_(self):
            assert False

        def _apply_unitary_(self, args):
            assert False

        def _unitary_(self):
            assert False

    class Yes2(EmptyOp):
        def _has_unitary_(self):
            return NotImplemented

        def _decompose_(self):
            return []

        def _apply_unitary_(self, args):
            assert False

        def _unitary_(self):
            assert False

    class Yes3(EmptyOp):
        def _has_unitary_(self):
            return NotImplemented

        def _decompose_(self):
            return NotImplemented

        def _apply_unitary_(self, args):
            return args.target_tensor

        def _unitary_(self):
            assert False

    class Yes4(EmptyOp):
        def _has_unitary_(self):
            return NotImplemented

        def _decompose_(self):
            return NotImplemented

        def _apply_unitary_(self, args):
            return NotImplemented

        def _unitary_(self):
            return np.array([[1]])

    assert cirq.has_unitary(Yes1())
    assert cirq.has_unitary(Yes2())
    assert cirq.has_unitary(Yes3())
    assert cirq.has_unitary(Yes4())


class EmptyOp(cirq.Operation):
    """A trivial operation that will be recognized as `_apply_unitary_`-able."""

    @property
    def qubits(self):
        # coverage: ignore
        return ()

    def with_qubits(self, *new_qubits):
        # coverage: ignore
        return self

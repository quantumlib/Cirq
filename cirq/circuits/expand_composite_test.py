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

"""Tests for the expand composite optimization pass."""

from cirq.circuits import (
    Circuit,
    DropEmptyMoments,
    ExpandComposite,
    InsertStrategy,
)
from cirq.extension import Extensions
from cirq.ops import (
    CNOT, CNotGate, CompositeGate, CZ, HGate, SWAP, X, Y, Z, ISWAP,
    NamedQubit, QubitId)


def assert_equal_mod_empty(expected, actual):
    drop_empty = DropEmptyMoments()
    drop_empty.optimize_circuit(actual)
    if expected != actual:
        # coverage: ignore
        print('EXPECTED')
        print(expected)
        print('ACTUAL')
        print(actual)
    assert expected == actual


def test_empty_circuit():
    circuit = Circuit()
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(Circuit(), circuit)


def test_empty_moment():
    circuit = Circuit([])
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(Circuit([]), circuit)


def test_ignore_non_composite():
    q0, q1 = QubitId(), QubitId()
    circuit = Circuit()
    circuit.append([X(q0), Y(q1), CZ(q0, q1), Z(q0)])
    expected = Circuit(circuit.moments)
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(expected, circuit)


def test_composite_default():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = Circuit()
    circuit.append(cnot)
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5])
    assert_equal_mod_empty(expected, circuit)


def test_multiple_composite_default():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = Circuit()
    circuit.append([cnot, cnot])
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit()
    decomp = [Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5]
    expected.append([decomp, decomp])
    assert_equal_mod_empty(expected, circuit)


def test_mix_composite_non_composite():
    q0, q1 = QubitId(), QubitId()

    actual = Circuit.from_ops(X(q0), CNOT(q0, q1), X(q1))
    opt = ExpandComposite()
    opt.optimize_circuit(actual)

    expected = Circuit.from_ops(X(q0),
                                Y(q1) ** -0.5,
                                CZ(q0, q1),
                                Y(q1) ** 0.5,
                                X(q1),
                                strategy=InsertStrategy.NEW)
    assert_equal_mod_empty(expected, actual)


def test_recursive_composite():
    q0, q1 = QubitId(), QubitId()
    swap = SWAP(q0, q1)
    circuit = Circuit()
    circuit.append(swap)

    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit().from_ops(Y(q1) ** -0.5,
                                  CZ(q0, q1),
                                  Y(q1) ** 0.5,
                                  Y(q0) ** -0.5,
                                  CZ(q1, q0),
                                  Y(q0) ** 0.5,
                                  Y(q1) ** -0.5,
                                  CZ(q0, q1),
                                  Y(q1) ** 0.5)
    assert_equal_mod_empty(expected, circuit)


def test_decompose_returns_not_flat_op_tree():
    class DummyGate(CompositeGate):
        def default_decompose(self, qubits):
            q0, = qubits
            # Yield a tuple of gates instead of yielding a gate
            yield X(q0),

    q0 = QubitId()
    circuit = Circuit.from_ops(DummyGate()(q0))

    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit().from_ops(X(q0))
    assert_equal_mod_empty(expected, circuit)


def test_decompose_returns_deep_op_tree():
    class DummyGate(CompositeGate):
        def default_decompose(self, qubits):
            q0, q1 = qubits
            # Yield a tuple
            yield ((X(q0), Y(q0)), Z(q0))
            # Yield nested lists
            yield [X(q0), [Y(q0), Z(q0)]]
            def generator(depth):
                if depth <= 0:
                    yield CZ(q0, q1), Y(q0)
                else:
                    yield X(q0), generator(depth - 1)
                    yield Z(q0)
            # Yield nested generators
            yield generator(2)

    q0, q1 = QubitId(), QubitId()
    circuit = Circuit.from_ops(DummyGate()(q0, q1))

    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit().from_ops(X(q0), Y(q0), Z(q0),  # From tuple
                                  X(q0), Y(q0), Z(q0),  # From nested lists
                                  # From nested generators
                                  X(q0), X(q0),
                                  CZ(q0, q1), Y(q0),
                                  Z(q0), Z(q0))
    assert_equal_mod_empty(expected, circuit)


class OtherCNot(CNotGate):

    def default_decompose(self, qubits):
        c, t = qubits
        yield Z(c)
        yield Y(t)**-0.5
        yield CZ(c, t)
        yield Y(t)**0.5
        yield Z(c)

def test_nonrecursive_expansion():
    qubits = [NamedQubit(s) for s in 'xy']
    stopper = lambda op: (op.gate == ISWAP)
    expander = ExpandComposite(stopper=stopper)
    unexpanded_circuit = Circuit.from_ops(ISWAP(*qubits))

    circuit = unexpanded_circuit.copy_moments()
    expander.optimize_circuit(circuit)
    assert circuit == unexpanded_circuit

    stopper = lambda op: isinstance(op.gate, (CNotGate, HGate))
    expander = ExpandComposite(stopper=stopper)
    circuit = unexpanded_circuit.copy_moments()
    expander.optimize_circuit(circuit)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
x: ───@───H───X───S───X───S^-1───H───@───
      │       │       │              │
y: ───X───────@───────@──────────────X───
    """.strip()
    assert actual_text_diagram == expected_text_diagram



def test_composite_extension_overrides():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = Circuit()
    circuit.append(cnot)
    opt = ExpandComposite(composite_gate_extension=Extensions({
        CompositeGate: {CNotGate: lambda e: OtherCNot()}
    }))
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([Z(q0), Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5, Z(q0)])
    assert_equal_mod_empty(expected, circuit)


def test_recursive_composite_extension_overrides():
    q0, q1 = QubitId(), QubitId()
    swap = SWAP(q0, q1)
    circuit = Circuit()
    circuit.append(swap)
    opt = ExpandComposite(composite_gate_extension=Extensions({
        CompositeGate: {CNotGate: lambda e: OtherCNot()}
    }))
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([Z(q0), Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5, Z(q0)])
    expected.append([Z(q1), Y(q0) ** -0.5, CZ(q1, q0), Y(q0) ** 0.5, Z(q1)],
                    strategy=InsertStrategy.INLINE)
    expected.append([Z(q0), Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5, Z(q0)],
                    strategy=InsertStrategy.INLINE)
    assert_equal_mod_empty(expected, circuit)

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
import cirq
from cirq.ops import CNOT, CZ, QubitId, SWAP, X, Y, Z


def assert_equal_mod_empty(expected, actual):
    drop_empty = cirq.DropEmptyMoments()
    drop_empty.optimize_circuit(actual)
    if expected != actual:
        # coverage: ignore
        print('EXPECTED')
        print(expected)
        print('ACTUAL')
        print(actual)
    assert expected == actual


def test_empty_circuit():
    circuit = cirq.Circuit()
    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(cirq.Circuit(), circuit)


def test_empty_moment():
    circuit = cirq.Circuit([])
    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(cirq.Circuit([]), circuit)


def test_ignore_non_composite():
    q0, q1 = QubitId(), QubitId()
    circuit = cirq.Circuit()
    circuit.append([X(q0), Y(q1), CZ(q0, q1), Z(q0)])
    expected = circuit.copy()
    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(expected, circuit)


def test_composite_default():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = cirq.Circuit()
    circuit.append(cnot)
    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = cirq.Circuit()
    expected.append([Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5])
    assert_equal_mod_empty(expected, circuit)


def test_multiple_composite_default():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = cirq.Circuit()
    circuit.append([cnot, cnot])
    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = cirq.Circuit()
    decomp = [Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5]
    expected.append([decomp, decomp])
    assert_equal_mod_empty(expected, circuit)


def test_mix_composite_non_composite():
    q0, q1 = QubitId(), QubitId()

    actual = cirq.Circuit.from_ops(X(q0), CNOT(q0, q1), X(q1))
    opt = cirq.ExpandComposite()
    opt.optimize_circuit(actual)

    expected = cirq.Circuit.from_ops(X(q0),
                                     Y(q1) ** -0.5,
                                     CZ(q0, q1),
                                     Y(q1) ** 0.5,
                                     X(q1),
                                     strategy=cirq.InsertStrategy.NEW)
    assert_equal_mod_empty(expected, actual)


def test_recursive_composite():
    q0, q1 = QubitId(), QubitId()
    swap = SWAP(q0, q1)
    circuit = cirq.Circuit()
    circuit.append(swap)

    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = cirq.Circuit().from_ops(Y(q1) ** -0.5,
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
    class DummyGate(cirq.Gate, cirq.CompositeGate):
        def default_decompose(self, qubits):
            q0, = qubits
            # Yield a tuple of gates instead of yielding a gate
            yield X(q0),

    q0 = QubitId()
    circuit = cirq.Circuit.from_ops(DummyGate()(q0))

    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = cirq.Circuit().from_ops(X(q0))
    assert_equal_mod_empty(expected, circuit)


def test_decompose_returns_deep_op_tree():
    class DummyGate(cirq.Gate, cirq.CompositeGate):
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
    circuit = cirq.Circuit.from_ops(DummyGate()(q0, q1))

    opt = cirq.ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = cirq.Circuit().from_ops(X(q0), Y(q0), Z(q0),  # From tuple
                                       X(q0), Y(q0), Z(q0),  # From nested lists
                                       # From nested generators
                                       X(q0), X(q0),
                                       CZ(q0, q1), Y(q0),
                                       Z(q0), Z(q0))
    assert_equal_mod_empty(expected, circuit)


class OtherCNot(cirq.CNotGate):

    def default_decompose(self, qubits):
        c, t = qubits
        yield Z(c)
        yield Y(t)**-0.5
        yield CZ(c, t)
        yield Y(t)**0.5
        yield Z(c)


def test_nonrecursive_expansion():
    qubits = [cirq.NamedQubit(s) for s in 'xy']
    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and
                            op.gate == cirq.ISWAP)
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    unexpanded_circuit = cirq.Circuit.from_ops(cirq.ISWAP(*qubits))

    circuit = unexpanded_circuit.__copy__()
    expander.optimize_circuit(circuit)
    assert circuit == unexpanded_circuit

    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and
                            isinstance(op.gate, (cirq.CNotGate, cirq.HGate)))
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    circuit = unexpanded_circuit.__copy__()
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
    circuit = cirq.Circuit()
    circuit.append(cnot)
    ext = cirq.Extensions()
    ext.add_cast(cirq.CompositeGate, cirq.CNotGate, lambda _: OtherCNot())
    opt = cirq.ExpandComposite(composite_gate_extension=ext)
    opt.optimize_circuit(circuit)
    expected = cirq.Circuit()
    expected.append([Z(q0), Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5, Z(q0)])
    assert_equal_mod_empty(expected, circuit)

def test_recursive_composite_extension_overrides():
    q0, q1 = QubitId(), QubitId()
    swap = SWAP(q0, q1)
    circuit = cirq.Circuit()
    circuit.append(swap)
    ext = cirq.Extensions()
    ext.add_cast(cirq.CompositeGate, cirq.CNotGate, lambda _: OtherCNot())
    opt = cirq.ExpandComposite(composite_gate_extension=ext)
    opt.optimize_circuit(circuit)
    expected = cirq.Circuit()
    expected.append([Z(q0), Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5, Z(q0)])
    expected.append([Z(q1), Y(q0) ** -0.5, CZ(q1, q0), Y(q0) ** 0.5, Z(q1)],
                    strategy=cirq.InsertStrategy.INLINE)
    expected.append([Z(q0), Y(q1) ** -0.5, CZ(q0, q1), Y(q1) ** 0.5, Z(q0)],
                    strategy=cirq.InsertStrategy.INLINE)
    assert_equal_mod_empty(expected, circuit)

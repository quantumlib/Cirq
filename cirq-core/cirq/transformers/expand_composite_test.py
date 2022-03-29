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

"""Tests for the expand composite transformer pass."""

import cirq


def assert_equal_mod_empty(expected, actual):
    actual = cirq.drop_empty_moments(actual)
    cirq.testing.assert_same_circuits(actual, expected)


def test_empty_circuit():
    circuit = cirq.Circuit()
    circuit = cirq.expand_composite(circuit)
    assert_equal_mod_empty(cirq.Circuit(), circuit)


def test_empty_moment():
    circuit = cirq.Circuit([])
    circuit = cirq.expand_composite(circuit)
    assert_equal_mod_empty(cirq.Circuit([]), circuit)


def test_ignore_non_composite():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append([cirq.X(q0), cirq.Y(q1), cirq.CZ(q0, q1), cirq.Z(q0)])
    expected = circuit.copy()
    circuit = cirq.expand_composite(circuit)
    assert_equal_mod_empty(expected, circuit)


def test_composite_default():
    q0, q1 = cirq.LineQubit.range(2)
    cnot = cirq.CNOT(q0, q1)
    circuit = cirq.Circuit()
    circuit.append(cnot)
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit()
    expected.append([cirq.Y(q1) ** -0.5, cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5])
    assert_equal_mod_empty(expected, circuit)


def test_multiple_composite_default():
    q0, q1 = cirq.LineQubit.range(2)
    cnot = cirq.CNOT(q0, q1)
    circuit = cirq.Circuit()
    circuit.append([cnot, cnot])
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit()
    decomp = [cirq.Y(q1) ** -0.5, cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5]
    expected.append([decomp, decomp])
    assert_equal_mod_empty(expected, circuit)


def test_mix_composite_non_composite():
    q0, q1 = cirq.LineQubit.range(2)

    circuit = cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.X(q1))
    circuit = cirq.expand_composite(circuit)

    expected = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q1) ** -0.5,
        cirq.CZ(q0, q1),
        cirq.Y(q1) ** 0.5,
        cirq.X(q1),
        strategy=cirq.InsertStrategy.NEW,
    )
    assert_equal_mod_empty(expected, circuit)


def test_recursive_composite():
    q0, q1 = cirq.LineQubit.range(2)
    swap = cirq.SWAP(q0, q1)
    circuit = cirq.Circuit()
    circuit.append(swap)

    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit(
        cirq.Y(q1) ** -0.5,
        cirq.CZ(q0, q1),
        cirq.Y(q1) ** 0.5,
        cirq.Y(q0) ** -0.5,
        cirq.CZ(q1, q0),
        cirq.Y(q0) ** 0.5,
        cirq.Y(q1) ** -0.5,
        cirq.CZ(q0, q1),
        cirq.Y(q1) ** 0.5,
    )
    assert_equal_mod_empty(expected, circuit)


def test_decompose_returns_not_flat_op_tree():
    class DummyGate(cirq.SingleQubitGate):
        def _decompose_(self, qubits):
            (q0,) = qubits
            # Yield a tuple of gates instead of yielding a gate
            yield cirq.X(q0),

    q0 = cirq.NamedQubit('q0')
    circuit = cirq.Circuit(DummyGate()(q0))

    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit(cirq.X(q0))
    assert_equal_mod_empty(expected, circuit)


def test_decompose_returns_deep_op_tree():
    class DummyGate(cirq.testing.TwoQubitGate):
        def _decompose_(self, qubits):
            q0, q1 = qubits
            # Yield a tuple
            yield ((cirq.X(q0), cirq.Y(q0)), cirq.Z(q0))
            # Yield nested lists
            yield [cirq.X(q0), [cirq.Y(q0), cirq.Z(q0)]]

            def generator(depth):
                if depth <= 0:
                    yield cirq.CZ(q0, q1), cirq.Y(q0)
                else:
                    yield cirq.X(q0), generator(depth - 1)
                    yield cirq.Z(q0)

            # Yield nested generators
            yield generator(2)

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(DummyGate()(q0, q1))

    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q0),  # From tuple
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q0),  # From nested lists
        # From nested generators
        cirq.X(q0),
        cirq.X(q0),
        cirq.CZ(q0, q1),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.Z(q0),
    )
    assert_equal_mod_empty(expected, circuit)


def test_non_recursive_expansion():
    qubits = [cirq.NamedQubit(s) for s in 'xy']
    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and op.gate == cirq.ISWAP)
    unexpanded_circuit = cirq.Circuit(cirq.ISWAP(*qubits))

    circuit = cirq.expand_composite(unexpanded_circuit, no_decomp=no_decomp)
    assert circuit == unexpanded_circuit

    no_decomp = lambda op: (
        isinstance(op, cirq.GateOperation)
        and isinstance(op.gate, (cirq.CNotPowGate, cirq.HPowGate))
    )
    circuit = cirq.expand_composite(unexpanded_circuit, no_decomp=no_decomp)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
x: ───@───H───X───S───X───S^-1───H───@───
      │       │       │              │
y: ───X───────@───────@──────────────X───
    """.strip()
    assert actual_text_diagram == expected_text_diagram


def test_do_not_decompose_no_compile():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(q0, q1).with_tags("no_compile"))
    context = cirq.TransformerContext(tags_to_ignore=("no_compile",))
    assert_equal_mod_empty(c, cirq.expand_composite(c, context=context))


def test_expands_composite_recursively_preserving_structur():
    q = cirq.LineQubit.range(2)
    c_nested = cirq.FrozenCircuit(
        cirq.SWAP(*q[:2]), cirq.SWAP(*q[:2]).with_tags("ignore"), cirq.SWAP(*q[:2])
    )
    c_nested_expanded = cirq.FrozenCircuit(
        [cirq.CNOT(*q), cirq.CNOT(*q[::-1]), cirq.CNOT(*q)],
        cirq.SWAP(*q[:2]).with_tags("ignore"),
        [cirq.CNOT(*q), cirq.CNOT(*q[::-1]), cirq.CNOT(*q)],
    )
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                c_nested,
                cirq.CircuitOperation(c_nested).repeat(5).with_tags("ignore"),
                cirq.CircuitOperation(c_nested).repeat(6).with_tags("preserve_tag"),
                cirq.CircuitOperation(c_nested).repeat(7),
                c_nested,
            )
        )
        .repeat(4)
        .with_tags("ignore"),
        c_nested,
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                c_nested,
                cirq.CircuitOperation(c_nested).repeat(5).with_tags("ignore"),
                cirq.CircuitOperation(c_nested).repeat(6).with_tags("preserve_tag"),
                cirq.CircuitOperation(c_nested).repeat(7),
                c_nested,
            )
        )
        .repeat(5)
        .with_tags("preserve_tag"),
        c_nested,
    )
    c_expected = cirq.Circuit(
        c_nested_expanded,
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                c_nested,
                cirq.CircuitOperation(c_nested).repeat(5).with_tags("ignore"),
                cirq.CircuitOperation(c_nested).repeat(6).with_tags("preserve_tag"),
                cirq.CircuitOperation(c_nested).repeat(7),
                c_nested,
            )
        )
        .repeat(4)
        .with_tags("ignore"),
        c_nested_expanded,
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                c_nested_expanded,
                cirq.CircuitOperation(c_nested).repeat(5).with_tags("ignore"),
                cirq.CircuitOperation(c_nested_expanded).repeat(6).with_tags("preserve_tag"),
                cirq.CircuitOperation(c_nested_expanded).repeat(7),
                c_nested_expanded,
            )
        )
        .repeat(5)
        .with_tags("preserve_tag"),
        c_nested_expanded,
    )

    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_expanded = cirq.expand_composite(
        c_orig, no_decomp=lambda op: op.gate == cirq.CNOT, context=context
    )
    cirq.testing.assert_same_circuits(c_expanded, c_expected)

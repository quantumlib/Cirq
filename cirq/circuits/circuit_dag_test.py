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

import networkx

import cirq


def test_wrapper_eq():
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.CircuitDag.make_node(cirq.X(q0)))
    eq.add_equality_group(cirq.CircuitDag.make_node(cirq.X(q0)))
    eq.add_equality_group(cirq.CircuitDag.make_node(cirq.Y(q0)))
    eq.add_equality_group(cirq.CircuitDag.make_node(cirq.X(q1)))


def test_wrapper_cmp():
    u0 = cirq.Unique(0)
    u1 = cirq.Unique(1)
    # The ordering of Unique instances is unpredictable
    u0, u1 = (u1, u0) if u1 < u0 else (u0, u1)
    assert u0 == u0
    assert u0 != u1
    assert u0 <  u1
    assert u1 >  u0
    assert u0 <= u0
    assert u0 <= u1
    assert u0 >= u0
    assert u1 >= u0


@cirq.testing.only_test_in_python3
def test_wrapper_cmp_failure():
    with pytest.raises(TypeError):
        _ = object() < cirq.Unique(1)
    with pytest.raises(TypeError):
        _ = cirq.Unique(1) < object()


def test_wrapper_repr():
    q0 = cirq.LineQubit(0)

    node = cirq.CircuitDag.make_node(cirq.X(q0))
    assert (repr(node) ==
            'cirq.Unique(' + str(id(node)) + ', cirq.X.on(cirq.LineQubit(0)))')


def test_init():
    dag = cirq.CircuitDag()
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert list(dag.nodes()) == []
    assert list(dag.edges()) == []


def test_append():
    q0 = cirq.LineQubit(0)
    dag = cirq.CircuitDag()
    dag.append(cirq.X(q0))
    dag.append(cirq.Y(q0))
    print(dag.edges())
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert len(dag.nodes()) == 2
    assert ([(n1.val, n2.val) for n1, n2 in dag.edges()] ==
            [(cirq.X(q0), cirq.Y(q0))])


def test_two_identical_ops():
    q0 = cirq.LineQubit(0)
    dag = cirq.CircuitDag()
    dag.append(cirq.X(q0))
    dag.append(cirq.Y(q0))
    dag.append(cirq.X(q0))
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert len(dag.nodes()) == 3
    assert (set((n1.val, n2.val) for n1, n2 in dag.edges()) ==
            {(cirq.X(q0), cirq.Y(q0)),
             (cirq.X(q0), cirq.X(q0)),
             (cirq.Y(q0), cirq.X(q0))})


def test_from_ops():
    q0 = cirq.LineQubit(0)
    dag = cirq.CircuitDag.from_ops(
        cirq.X(q0),
        cirq.Y(q0))
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert len(dag.nodes()) == 2
    assert ([(n1.val, n2.val) for n1, n2 in dag.edges()] ==
            [(cirq.X(q0), cirq.Y(q0))])


def test_from_circuit():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0))
    dag = cirq.CircuitDag.from_circuit(circuit)
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert len(dag.nodes()) == 2
    assert ([(n1.val, n2.val) for n1, n2 in dag.edges()] ==
            [(cirq.X(q0), cirq.Y(q0))])


def test_from_circuit_with_device():
    q0 = cirq.GridQubit(5, 5)
    circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0),
        device=cirq.google.Bristlecone)
    dag = cirq.CircuitDag.from_circuit(circuit)
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert dag.device == circuit.device
    assert len(dag.nodes()) == 2
    assert ([(n1.val, n2.val) for n1, n2 in dag.edges()] ==
            [(cirq.X(q0), cirq.Y(q0))])


def test_to_empty_circuit():
    circuit = cirq.Circuit()
    dag = cirq.CircuitDag.from_circuit(circuit)
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert circuit == dag.to_circuit()


def test_to_circuit():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0))
    dag = cirq.CircuitDag.from_circuit(circuit)

    assert networkx.dag.is_directed_acyclic_graph(dag)
    # Only one possible output circuit for this simple case
    assert circuit == dag.to_circuit()

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.to_unitary_matrix(),
        dag.to_circuit().to_unitary_matrix(),
        atol=1e-7)


def test_equality():
    q0, q1 = cirq.LineQubit.range(2)
    circuit1 = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q1),
        cirq.CZ(q0, q1),
        cirq.X(q1),
        cirq.Y(q1),
        cirq.Z(q0),
    )
    circuit2 = cirq.Circuit.from_ops(
        cirq.Z(q1),
        cirq.X(q0),
        cirq.Y(q0),
        cirq.CZ(q0, q1),
        cirq.Z(q0),
        cirq.X(q1),
        cirq.Y(q1),
    )
    circuit3 = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q1),
        cirq.CZ(q0, q1),
        cirq.X(q1),
        cirq.Y(q1),
        cirq.Z(q0) ** 0.5,
    )
    circuit4 = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q1),
        cirq.CZ(q0, q1),
        cirq.X(q1),
        cirq.Y(q1),
    )

    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(
        lambda: cirq.CircuitDag.from_circuit(circuit1),
        lambda: cirq.CircuitDag.from_circuit(circuit2),
    )
    eq.add_equality_group(
        cirq.CircuitDag.from_circuit(circuit3),
    )
    eq.add_equality_group(
        cirq.CircuitDag.from_circuit(circuit4),
    )


def test_larger_circuit():
    q0, q1, q2, q3 = cirq.google.Bristlecone.col(5)[:4]
    # This circuit does not have CZ gates on adjacent qubits because the order
    # dag.to_circuit() would append them is non-deterministic.
    circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.CZ(q1, q2),
        cirq.CZ(q0, q1),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.CZ(q1, q2),
        cirq.X(q0),
        cirq.Y(q0),
        cirq.CZ(q0, q1),
        cirq.T(q3),
        strategy=cirq.InsertStrategy.EARLIEST,
        device=cirq.google.Bristlecone)

    dag = cirq.CircuitDag.from_circuit(circuit)

    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert circuit.device == dag.to_circuit().device
    # Operation order within a moment is non-deterministic
    # but text diagrams still look the same.
    desired = """
(0, 5): ───X───@───Y───Z───X───Y───@───
               │                   │
(1, 5): ───@───@───@───────────────@───
           │       │
(2, 5): ───@───────@───────────────────

(3, 5): ───T───────────────────────────
"""
    cirq.testing.assert_has_diagram(circuit, desired)
    cirq.testing.assert_has_diagram(dag.to_circuit(), desired)

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.to_unitary_matrix(),
        dag.to_circuit().to_unitary_matrix(),
        atol=1e-7)

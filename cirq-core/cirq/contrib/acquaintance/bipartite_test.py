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

import itertools

import pytest

import cirq
import cirq.contrib.acquaintance as cca


circuit_diagrams = {
    (
        'undecomposed',
        cca.BipartiteGraphType.COMPLETE,
        1,
    ): """
0: ───K_{1, 1}:(0, 0)↦(0, 0)───
      │
1: ───K_{1, 1}:(1, 0)↦(1, 0)───
""",
    (
        'undecomposed',
        cca.BipartiteGraphType.COMPLETE,
        2,
    ): """
0: ───K_{2, 2}:(0, 0)↦(0, 0)───
      │
1: ───K_{2, 2}:(0, 1)↦(0, 1)───
      │
2: ───K_{2, 2}:(1, 0)↦(1, 0)───
      │
3: ───K_{2, 2}:(1, 1)↦(1, 1)───
""",
    (
        'undecomposed',
        cca.BipartiteGraphType.COMPLETE,
        3,
    ): """
0: ───K_{3, 3}:(0, 0)↦(0, 0)───
      │
1: ───K_{3, 3}:(0, 1)↦(0, 1)───
      │
2: ───K_{3, 3}:(0, 2)↦(0, 2)───
      │
3: ───K_{3, 3}:(1, 0)↦(1, 0)───
      │
4: ───K_{3, 3}:(1, 1)↦(1, 1)───
      │
5: ───K_{3, 3}:(1, 2)↦(1, 2)───
""",
    (
        'undecomposed',
        cca.BipartiteGraphType.COMPLETE,
        4,
    ): """
0: ───K_{4, 4}:(0, 0)↦(0, 0)───
      │
1: ───K_{4, 4}:(0, 1)↦(0, 1)───
      │
2: ───K_{4, 4}:(0, 2)↦(0, 2)───
      │
3: ───K_{4, 4}:(0, 3)↦(0, 3)───
      │
4: ───K_{4, 4}:(1, 0)↦(1, 0)───
      │
5: ───K_{4, 4}:(1, 1)↦(1, 1)───
      │
6: ───K_{4, 4}:(1, 2)↦(1, 2)───
      │
7: ───K_{4, 4}:(1, 3)↦(1, 3)───
""",
    (
        'undecomposed',
        cca.BipartiteGraphType.MATCHING,
        1,
    ): """
0: ───Matching:(0, 0)↦(0, 0)───
      │
1: ───Matching:(1, 0)↦(1, 0)───
""",
    (
        'undecomposed',
        cca.BipartiteGraphType.MATCHING,
        2,
    ): """
0: ───Matching:(0, 0)↦(0, 1)───
      │
1: ───Matching:(0, 1)↦(0, 0)───
      │
2: ───Matching:(1, 0)↦(1, 1)───
      │
3: ───Matching:(1, 1)↦(1, 0)───
""",
    (
        'undecomposed',
        cca.BipartiteGraphType.MATCHING,
        3,
    ): """
0: ───Matching:(0, 0)↦(0, 2)───
      │
1: ───Matching:(0, 1)↦(0, 1)───
      │
2: ───Matching:(0, 2)↦(0, 0)───
      │
3: ───Matching:(1, 0)↦(1, 2)───
      │
4: ───Matching:(1, 1)↦(1, 1)───
      │
5: ───Matching:(1, 2)↦(1, 0)───
""",
    (
        'undecomposed',
        cca.BipartiteGraphType.MATCHING,
        4,
    ): """
0: ───Matching:(0, 0)↦(0, 3)───
      │
1: ───Matching:(0, 1)↦(0, 2)───
      │
2: ───Matching:(0, 2)↦(0, 1)───
      │
3: ───Matching:(0, 3)↦(0, 0)───
      │
4: ───Matching:(1, 0)↦(1, 3)───
      │
5: ───Matching:(1, 1)↦(1, 2)───
      │
6: ───Matching:(1, 2)↦(1, 1)───
      │
7: ───Matching:(1, 3)↦(1, 0)───
""",
    (
        'decomposed',
        cca.BipartiteGraphType.COMPLETE,
        1,
    ): """
0: ───█───
      │
1: ───█───
""",
    (
        'decomposed',
        cca.BipartiteGraphType.COMPLETE,
        2,
    ): """
0: ─────────────█───0↦1───────█───0↦1─────────────
                │   │         │   │
1: ───█───0↦1───█───1↦0───█───█───1↦0───█───0↦1───
      │   │               │             │   │
2: ───█───1↦0───█───0↦1───█───█───0↦1───█───1↦0───
                │   │         │   │
3: ─────────────█───1↦0───────█───1↦0─────────────
""",
    ('decomposed', cca.BipartiteGraphType.COMPLETE, 3):
    # pylint: disable=line-too-long
    """
0: ───────────────────────█───0↦1───────────────────────────█───0↦1───────────────────────
                          │   │                             │   │
1: ─────────────█───0↦1───█───1↦0───█───0↦1───────█───0↦1───█───1↦0───█───0↦1─────────────
                │   │               │   │         │   │               │   │
2: ───█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───█───1↦0───█───0↦1───█───1↦0───█───0↦1───
      │   │               │   │               │             │   │               │   │
3: ───█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───█───0↦1───█───1↦0───█───0↦1───█───1↦0───
                │   │               │   │         │   │               │   │
4: ─────────────█───1↦0───█───0↦1───█───1↦0───────█───1↦0───█───0↦1───█───1↦0─────────────
                          │   │                             │   │
5: ───────────────────────█───1↦0───────────────────────────█───1↦0───────────────────────

""",
    (
        'decomposed',
        cca.BipartiteGraphType.COMPLETE,
        4,
    ): """
0: ─────────────────────────────────█───0↦1───────────────────────────────────────────────█───0↦1─────────────────────────────────
                                    │   │                                                 │   │
1: ───────────────────────█───0↦1───█───1↦0───█───0↦1───────────────────────────█───0↦1───█───1↦0───█───0↦1───────────────────────
                          │   │               │   │                             │   │               │   │
2: ─────────────█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───0↦1───────█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───0↦1─────────────
                │   │               │   │               │   │         │   │               │   │               │   │
3: ───█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───0↦1───
      │   │               │   │               │   │               │             │   │               │   │               │   │
4: ───█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───█───0↦1───█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───1↦0───
                │   │               │   │               │   │         │   │               │   │               │   │
5: ─────────────█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───1↦0───────█───1↦0───█───0↦1───█───1↦0───█───0↦1───█───1↦0─────────────
                          │   │               │   │                             │   │               │   │
6: ───────────────────────█───1↦0───█───0↦1───█───1↦0───────────────────────────█───1↦0───█───0↦1───█───1↦0───────────────────────
                                    │   │                                                 │   │
7: ─────────────────────────────────█───1↦0───────────────────────────────────────────────█───1↦0─────────────────────────────────

""",
    # pylint: enable=line-too-long
    (
        'decomposed',
        cca.BipartiteGraphType.MATCHING,
        1,
    ): """
0: ───█───
      │
1: ───█───
""",
    (
        'decomposed',
        cca.BipartiteGraphType.MATCHING,
        2,
    ): """
0: ───────0↦1───────
          │
1: ───█───1↦0───█───
      │         │
2: ───█───0↦1───█───
          │
3: ───────1↦0───────
""",
    (
        'decomposed',
        cca.BipartiteGraphType.MATCHING,
        3,
    ): """
0: ─────────────0↦1─────────────
                │
1: ───────0↦1───1↦0───0↦1───────
          │           │
2: ───█───1↦0───█─────1↦0───█───
      │         │           │
3: ───█───0↦1───█─────0↦1───█───
          │           │
4: ───────1↦0───0↦1───1↦0───────
                │
5: ─────────────1↦0─────────────
""",
    (
        'decomposed',
        cca.BipartiteGraphType.MATCHING,
        4,
    ): """
0: ───────────────────0↦1───────────────────
                      │
1: ─────────────0↦1───1↦0───0↦1─────────────
                │           │
2: ───────0↦1───1↦0───0↦1───1↦0───0↦1───────
          │           │           │
3: ───█───1↦0───█─────1↦0───█─────1↦0───█───
      │         │           │           │
4: ───█───0↦1───█─────0↦1───█─────0↦1───█───
          │           │           │
5: ───────1↦0───0↦1───1↦0───0↦1───1↦0───────
                │           │
6: ─────────────1↦0───0↦1───1↦0─────────────
                      │
7: ───────────────────1↦0───────────────────
""",
}


@pytest.mark.parametrize(
    'subgraph,part_size', itertools.product(cca.BipartiteGraphType, range(1, 5))
)
def test_circuit_diagrams(part_size, subgraph):
    qubits = cirq.LineQubit.range(2 * part_size)
    gate = cca.BipartiteSwapNetworkGate(subgraph, part_size)
    circuit = cirq.Circuit(gate(*qubits))
    diagram = circuit_diagrams['undecomposed', subgraph, part_size]
    cirq.testing.assert_has_diagram(circuit, diagram)

    no_decomp = lambda op: isinstance(
        op.gate, (cca.AcquaintanceOpportunityGate, cca.SwapPermutationGate)
    )
    circuit = cirq.expand_composite(circuit, no_decomp=no_decomp)
    diagram = circuit_diagrams['decomposed', subgraph, part_size]
    cirq.testing.assert_has_diagram(circuit, diagram)


def test_bad_args():
    gate = cca.BipartiteSwapNetworkGate(cca.BipartiteGraphType.COMPLETE, 2)
    qubits = cirq.LineQubit.range(4)
    gate.subgraph = 'not a subgraph'
    args = cirq.CircuitDiagramInfoArgs(
        known_qubits=None,
        known_qubit_count=None,
        use_unicode_characters=True,
        precision=3,
        label_map=None,
    )
    with pytest.raises(NotImplementedError):
        gate._circuit_diagram_info_(args)

    args.known_qubit_count = 3
    with pytest.raises(ValueError):
        gate._circuit_diagram_info_(args)

    with pytest.raises(ValueError):
        gate._decompose_(qubits[:3])

    gate.subgraph = 'unimplemented subgraph'
    with pytest.raises(NotImplementedError):
        gate._decompose_(qubits)

    args.known_qubit_count = None
    with pytest.raises(NotImplementedError):
        gate._circuit_diagram_info_(args)


def test_bipartite_swap_network_acquaintance_size():
    qubits = cirq.LineQubit.range(4)
    gate = cca.BipartiteSwapNetworkGate(cca.BipartiteGraphType.COMPLETE, 2)
    assert cca.get_acquaintance_size(gate(*qubits)) == 2


@pytest.mark.parametrize(
    'subgraph,part_size', itertools.product(cca.BipartiteGraphType, range(1, 3))
)
def test_repr(subgraph, part_size):
    gate = cca.BipartiteSwapNetworkGate(subgraph, part_size)
    cirq.testing.assert_equivalent_repr(gate)

    gate = cca.BipartiteSwapNetworkGate(subgraph, part_size, cirq.ZZ)
    cirq.testing.assert_equivalent_repr(gate)


@pytest.mark.parametrize(
    'subgraph,part_size', itertools.product(cca.BipartiteGraphType, range(1, 6))
)
def test_decomposition_permutation_consistency(part_size, subgraph):
    gate = cca.BipartiteSwapNetworkGate(subgraph, part_size)
    qubits = cirq.LineQubit.range(2 * part_size)
    mapping = {q: i for i, q in enumerate(qubits)}
    cca.update_mapping(mapping, gate._decompose_(qubits))
    permutation = gate.permutation()
    assert {qubits[i]: j for i, j in permutation.items()} == mapping

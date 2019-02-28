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

@pytest.mark.parametrize('graph_type', cca.double_bipartite.GraphType)
def test_graph_type_repr(graph_type):
    cirq.testing.assert_equivalent_repr(graph_type)

def double_bipartite_swap_network_gates(
        ns_left_qubits,
        ns_right_qubits = None,
        swap_gates = (cirq.SWAP,),
        shifteds = (None, True, False),
        balanceds = (True, False),
        acquaintance_graphs = ('NONE', 'BIPARTITE')):
    if ns_right_qubits is None:
        ns_right_qubits = ns_left_qubits
    return tuple(cca.DoubleBipartiteSwapNetworkGate(
            n_left_qubits * (2 if balanced else 1),
            n_right_qubits * (2 if balanced else 1),
            swap_gate=swap_gate,
            shifted=shifted,
            balanced=balanced,
            acquaintance_graph=acquaintance_graph) for
            (n_left_qubits, n_right_qubits, swap_gate,
             shifted, balanced, acquaintance_graph) in
            itertools.product(
                ns_left_qubits, ns_right_qubits, swap_gates,
                shifteds, balanceds, acquaintance_graphs))


@pytest.mark.parametrize('gate',
        double_bipartite_swap_network_gates(
            range(1, 7),
            swap_gates = (cirq.SWAP, cirq.ZZ)
        ))
def test_double_bipartite_swap_network_gate_repr(gate):
    cirq.testing.assert_equivalent_repr(gate)

@pytest.mark.parametrize('gate',
        double_bipartite_swap_network_gates(range(1, 7))
        )
def test_double_bipartite_swap_network_gate_permutation(gate):
    n_qubits = gate.n_left_qubits + gate.n_right_qubits
    cca.testing.assert_permutation_decomposition_equivalence(gate, n_qubits)

def test_unsupported_graph_types():
    gate = cca.DoubleBipartiteSwapNetworkGate(3, 3)
    gate.acquaintance_graph = 'unsupported graph'
    with pytest.raises(NotImplementedError):
        gate.permutation()

    qubits = cirq.LineQubit.range(6)
    with pytest.raises(NotImplementedError):
        tuple(gate._decompose_(qubits))

@pytest.mark.parametrize('gate',
        double_bipartite_swap_network_gates(range(1, 5)))
def test_double_bipartite_swap_network_gate_acquaintance_opps(gate):
    n_qubits = gate.n_left_qubits + gate.n_right_qubits
    qubits = cirq.LineQubit.range(n_qubits)
    strategy = cirq.Circuit.from_ops(gate(*qubits),
            device=cca.UnconstrainedAcquaintanceDevice)

    # actual_opps
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    actual_opps = cca.get_logical_acquaintance_opportunities(
        strategy, initial_mapping)

    # expected opps
    indices = {
        'left': tuple(range(gate.n_left_qubits)),
        'right': tuple(gate.n_left_qubits + i
            for i in range(gate.n_right_qubits))}
    index_pairs = {side: tuple(itertools.combinations(indices[side], 2))
        for side in indices}
    if gate.acquaintance_graph.name == 'NONE':
        expected_opps = set(frozenset(index_pair)
                for index_pair in itertools.chain(*index_pairs.values()))
        assert actual_opps == expected_opps
    elif gate.acquaintance_graph.name == 'BIPARTITE':
        pairs_of_pairs = itertools.product(*index_pairs.values())
        if gate.balanced:
            def parity(I):
                return sum(i + (i // 2) for i in I) % 2
            pairs_of_pairs = ((I, J) for I, J in pairs_of_pairs if
                    parity(I) == parity(J))
        expected_opps = set(frozenset(I + J) for I, J in pairs_of_pairs)
        assert set(len(opp) for opp in actual_opps) <= set([2, 3, 4])
        quartic_opps = set(opp for opp in actual_opps if len(opp) == 4)
        assert quartic_opps == expected_opps
    else:
        raise NotImplementedError()

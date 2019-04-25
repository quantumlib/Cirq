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

import random

import pytest

import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca


def test_swap_permutation_gate():
    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and
                            op.gate == cirq.SWAP)
    a, b = cirq.NamedQubit('a'), cirq.NamedQubit('b')
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    gate = cca.SwapPermutationGate()
    assert gate.num_qubits() == 2
    circuit = cirq.Circuit.from_ops(gate(a, b))
    expander(circuit)
    assert tuple(circuit.all_operations()) == (cirq.SWAP(a, b),)


    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and
                            op.gate == cirq.CZ)
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    circuit = cirq.Circuit.from_ops(cca.SwapPermutationGate(cirq.CZ)(a, b))
    expander(circuit)
    assert tuple(circuit.all_operations()) == (cirq.CZ(a, b),)


def test_validate_permutation_errors():
    validate_permutation = cca.PermutationGate.validate_permutation
    validate_permutation({})

    with pytest.raises(IndexError,
            message='key and value sets must be the same.'):
        validate_permutation({0: 2, 1: 3})

    with pytest.raises(IndexError,
            message='keys of the permutation must be non-negative.'):
        validate_permutation({-1: 0, 0: -1})

    with pytest.raises(IndexError, message='key is out of bounds.'):
        validate_permutation({0: 3, 3: 0}, 2)

    gate = cca.SwapPermutationGate()
    assert cirq.circuit_diagram_info(gate, default=None) is None


def test_diagram():
    gate = cca.SwapPermutationGate()
    a, b = cirq.NamedQubit('a'), cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops([gate(a, b)])
    actual_text_diagram = circuit.to_text_diagram()
    expected_text_diagram = """
a: ───0↦1───
      │
b: ───1↦0───
    """.strip()
    assert actual_text_diagram == expected_text_diagram


def test_update_mapping():
    gate = cca.SwapPermutationGate()
    a, b, c = (cirq.NamedQubit(s) for s in 'abc')
    mapping = {s: i for i, s in enumerate((a, b, c))}
    ops = [gate(a, b), gate(b, c)]
    cca.update_mapping(mapping, ops)
    assert mapping == {a: 1, b: 2, c: 0}


@pytest.mark.parametrize('n_elements,n_permuted',
    ((n_elements, random.randint(0, n_elements)) for
     n_elements in (random.randint(5, 20) for _ in range(20))))
def test_linear_permutation_gate(n_elements, n_permuted):
    qubits = cirq.LineQubit.range(n_elements)
    elements = tuple(range(n_elements))
    elements_to_permute = random.sample(elements, n_permuted)
    permuted_elements = random.sample(elements_to_permute, n_permuted)
    permutation = {e: p for e, p in
                   zip(elements_to_permute, permuted_elements)}
    cca.PermutationGate.validate_permutation(permutation, n_elements)
    gate = cca.LinearPermutationGate(n_elements, permutation)
    ct.assert_equivalent_repr(gate)
    assert gate.permutation() == permutation
    mapping = dict(zip(qubits, elements))
    for swap in cirq.flatten_op_tree(cirq.decompose_once_with_qubits(
            gate, qubits)):
        assert isinstance(swap, cirq.GateOperation)
        swap.gate.update_mapping(mapping, swap.qubits)
    for i in range(n_elements):
        p = permutation.get(elements[i], i)
        assert mapping.get(qubits[p], elements[i]) == i


def random_equal_permutations(n_perms, n_items, prob):
    indices_to_permute = [i for i in range(n_items) if random.random() <= prob]
    permuted_indices = random.sample(
            indices_to_permute, len(indices_to_permute))
    base_permutation = dict(zip(indices_to_permute, permuted_indices))
    fixed_indices = [i for i in range(n_items) if i not in base_permutation]
    permutations = []
    for _ in range(n_perms):
        permutation = base_permutation.copy()
        permutation.update(
                {i: i for i in fixed_indices if random.random() <= prob})
        permutations.append(permutation)
    return permutations


def random_permutation_equality_groups(
        n_groups, n_perms_per_group, n_items, prob):
    fingerprints = set()
    for _ in range(n_groups):
        perms = random_equal_permutations(n_perms_per_group, n_items, prob)
        perm = perms[0]
        fingerprint = tuple(perm.get(i, i) for i in range(n_items))
        if fingerprint not in fingerprints:
            yield perms
            fingerprints.add(fingerprint)


@pytest.mark.parametrize('permutation_sets',
    [random_permutation_equality_groups(5, 3, 10, 0.5)])
def test_linear_permutation_gate_equality(permutation_sets):
    swap_gates = [cirq.SWAP, cirq.CNOT]
    equals_tester = ct.EqualsTester()
    for swap_gate in swap_gates:
        for permutation_set in permutation_sets:
            equals_tester.add_equality_group(*(
                cca.LinearPermutationGate(10, permutation, swap_gate)
                for permutation in permutation_set))


def test_linear_permutation_gate_pow_not_implemented():
    permutation_gate = cca.LinearPermutationGate(3, {0: 1, 1: 2, 2: 0})

    assert permutation_gate.__pow__(0) is NotImplemented
    assert permutation_gate.__pow__(2) is NotImplemented
    assert permutation_gate.__pow__(-2) is NotImplemented
    assert permutation_gate.__pow__(0.5) is NotImplemented
    assert permutation_gate.__pow__(-0.5) is NotImplemented


@pytest.mark.parametrize('num_qubits,permutation', [(2, {
    0: 1,
    1: 0
}), (3, {
    0: 0,
    1: 1,
    2: 2
}), (3, {
    0: 1,
    1: 2,
    2: 0
}), (3, {
    0: 2,
    1: 0,
    2: 1
}), (4, {
    0: 3,
    1: 2,
    2: 1,
    3: 0
})])
def test_linear_permutation_gate_pow_identity(num_qubits, permutation):
    permutation_gate = cca.LinearPermutationGate(num_qubits, permutation)

    assert permutation_gate**1 == permutation_gate


@pytest.mark.parametrize('num_qubits,permutation,inverse', [(2, {
    0: 1,
    1: 0
}, {
    0: 1,
    1: 0
}), (3, {
    0: 0,
    1: 1,
    2: 2
}, {
    0: 0,
    1: 1,
    2: 2
}), (3, {
    0: 1,
    1: 2,
    2: 0
}, {
    0: 2,
    1: 0,
    2: 1
}), (3, {
    0: 2,
    1: 0,
    2: 1
}, {
    0: 1,
    1: 2,
    2: 0
}), (4, {
    0: 3,
    1: 2,
    2: 1,
    3: 0
}, {
    0: 3,
    1: 2,
    2: 1,
    3: 0
})])
def test_linear_permutation_gate_pow_inverse(num_qubits, permutation, inverse):
    permutation_gate = cca.LinearPermutationGate(num_qubits, permutation)
    inverse_gate = cca.LinearPermutationGate(num_qubits, inverse)

    assert permutation_gate**-1 == inverse_gate
    assert cirq.inverse(permutation_gate) == inverse_gate

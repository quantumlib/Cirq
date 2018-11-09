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

from random import sample, randint
from string import ascii_lowercase as alphabet

import pytest

import cirq
from cirq.contrib.acquaintance.permutation import (
        PermutationGate, SwapPermutationGate, LinearPermutationGate,
        update_mapping)

def test_swap_permutation_gate():
    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and
                            op.gate == cirq.SWAP)
    a, b = cirq.NamedQubit('a'), cirq.NamedQubit('b')
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    circuit = cirq.Circuit.from_ops(SwapPermutationGate()(a, b))
    expander(circuit)
    assert tuple(circuit.all_operations()) == (cirq.SWAP(a, b),)


    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and
                            op.gate == cirq.CZ)
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    circuit = cirq.Circuit.from_ops(SwapPermutationGate(cirq.CZ)(a, b))
    expander(circuit)
    assert tuple(circuit.all_operations()) == (cirq.CZ(a, b),)

def test_validate_permutation_errors():
    validate_permutation = PermutationGate.validate_permutation
    validate_permutation({})

    with pytest.raises(IndexError,
            message='key and value sets must be the same.'):
        validate_permutation({0: 2, 1: 3})

    with pytest.raises(IndexError,
            message='keys of the permutation must be non-negative.'):
        validate_permutation({-1: 0, 0: -1})

    with pytest.raises(IndexError, message='key is out of bounds.'):
        validate_permutation({0: 3, 3: 0}, 2)

    gate = SwapPermutationGate()
    assert cirq.circuit_diagram_info(gate, default=None) is None


def test_diagram():
    gate = SwapPermutationGate()
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
    gate = SwapPermutationGate()
    a, b, c = (cirq.NamedQubit(s) for s in 'abc')
    mapping = {s: i for i, s in enumerate((a, b, c))}
    ops = [gate(a, b), gate(b, c)]
    update_mapping(mapping, ops)
    assert mapping == {a: 1, b: 2, c: 0}


def test_linear_permutation_gate():
    for _ in range(20):
        n_elements = randint(5, 20)
        n_permuted = randint(0, n_elements)
        qubits = [cirq.NamedQubit(s) for s in alphabet[:n_elements]]
        elements = tuple(range(n_elements))
        elements_to_permute = sample(elements, n_permuted)
        permuted_elements = sample(elements_to_permute, n_permuted)
        permutation = {e: p for e, p in
                       zip(elements_to_permute, permuted_elements)}
        PermutationGate.validate_permutation(permutation, n_elements)
        gate = LinearPermutationGate(permutation)
        assert gate.permutation(n_elements) == permutation
        mapping = dict(zip(qubits, elements))
        for swap in cirq.flatten_op_tree(cirq.decompose_once_with_qubits(
                gate, qubits)):
            assert isinstance(swap, cirq.GateOperation)
            swap.gate.update_mapping(mapping, swap.qubits)
        for i in range(n_elements):
            p = permutation.get(elements[i], i)
            assert mapping.get(qubits[p], elements[i]) == i

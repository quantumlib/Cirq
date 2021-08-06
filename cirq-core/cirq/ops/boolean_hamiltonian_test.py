# Copyright 2021 The Cirq Developers
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
import math

import numpy as np
import pytest
import sympy.parsing.sympy_parser as sympy_parser

import cirq


@pytest.mark.parametrize(
    'boolean_str',
    [
        'x0',
        '~x0',
        'x0 ^ x1',
        'x0 & x1',
        'x0 | x1',
        'x0 & x1 & x2',
        'x0 & x1 & ~x2',
        'x0 & ~x1 & x2',
        'x0 & ~x1 & ~x2',
        '~x0 & x1 & x2',
        '~x0 & x1 & ~x2',
        '~x0 & ~x1 & x2',
        '~x0 & ~x1 & ~x2',
        'x0 ^ x1 ^ x2',
        'x0 | (x1 & x2)',
        'x0 & (x1 | x2)',
        '(x0 ^ x1 ^ x2) | (x2 ^ x3 ^ x4)',
        '(x0 ^ x2 ^ x4) | (x1 ^ x2 ^ x3)',
        'x0 & x1 & (x2 | x3)',
        'x0 & ~x2',
        '~x0 & x2',
        'x2 & ~x0',
        '~x2 & x0',
        '(x2 | x1) ^ x0',
    ],
)
def test_circuit(boolean_str):
    boolean_expr = sympy_parser.parse_expr(boolean_str)
    var_names = cirq.parameter_names(boolean_expr)
    qubits = [cirq.NamedQubit(name) for name in var_names]

    # We use Sympy to evaluate the expression:
    n = len(var_names)

    expected = []
    for binary_inputs in itertools.product([0, 1], repeat=n):
        subed_expr = boolean_expr
        for var_name, binary_input in zip(var_names, binary_inputs):
            subed_expr = subed_expr.subs(var_name, binary_input)
        expected.append(bool(subed_expr))

    # We build a circuit and look at its output state vector:
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*qubits))

    hamiltonian_gate = cirq.BooleanHamiltonian(
        {q.name: q for q in qubits}, [boolean_str], 0.1 * math.pi
    )
    assert hamiltonian_gate.with_qubits(*qubits) == hamiltonian_gate

    assert hamiltonian_gate.num_qubits() == n

    circuit.append(hamiltonian_gate)

    phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()
    actual = np.arctan2(phi.real, phi.imag) - math.pi / 2.0 > 0.0

    # Compare the two:
    np.testing.assert_array_equal(actual, expected)

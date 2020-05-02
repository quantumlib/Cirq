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
from typing import Optional, Dict, Sequence, Union, cast
from random import randint, random, sample, choice

import numpy as np
import pytest

import cirq
from cirq.testing.random_circuit import random_circuit, DEFAULT_GATE_DOMAIN


def test_random_circuit_errors():
    with pytest.raises(ValueError):
        random_circuit(randint(1, 10), randint(1, 10), -1)
    with pytest.raises(ValueError):
        random_circuit(randint(1, 10), randint(1, 10), 1.)

    with pytest.raises(ValueError):
        random_circuit(randint(1, 10), randint(1, 10), random(), gate_domain={})

    with pytest.raises(ValueError):
        random_circuit(0, randint(1, 10), random())

    with pytest.raises(ValueError):
        random_circuit((), randint(1, 10), random())


@pytest.mark.parametrize(
    'n_qubits,n_moments,op_density,gate_domain,pass_qubits',
    [(
        randint(1, 20),
        randint(1, 10),
        random(),
        (
            None
            if randint(0, 1)
            else dict(sample(tuple(DEFAULT_GATE_DOMAIN.items()),
                             randint(1, len(DEFAULT_GATE_DOMAIN))))
        ),
        choice((True, False))
    ) for _ in range(10)]
)
def test_random_circuit(n_qubits: Union[int, Sequence[cirq.Qid]],
                        n_moments: int,
                        op_density: float,
                        gate_domain: Optional[Dict[cirq.Gate, int]],
                        pass_qubits: bool):
    qubit_set = cirq.LineQubit.range(n_qubits)
    qubit_arg = qubit_set if pass_qubits else n_qubits
    circuit = random_circuit(qubit_arg, n_moments, op_density, gate_domain)
    if qubit_arg is qubit_set:
        assert circuit.all_qubits().issubset(qubit_set)
    assert len(circuit) == n_moments
    if gate_domain is None:
        gate_domain = DEFAULT_GATE_DOMAIN
    assert set(cast(cirq.GateOperation, op).gate
               for op in circuit.all_operations()
               ).issubset(gate_domain)


@pytest.mark.parametrize('seed', [randint(0, 2**32) for _ in range(10)])
def test_random_circuit_reproducible_with_seed(seed):
    wrappers = (lambda s: s, np.random.RandomState)
    circuits = [
        random_circuit(qubits=10,
                       n_moments=10,
                       op_density=0.7,
                       random_state=wrapper(seed))
        for wrapper in wrappers
        for _ in range(2)
    ]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*circuits)


def test_random_circuit_reproducible_between_runs():
    circuit = random_circuit(5, 8, 0.5, random_state=77)
    expected_diagram = """
                  ┌──┐
0: ────────────────S─────iSwap───────Y───X───
                         │
1: ───────────Y──────────iSwap───────Y───────

2: ─────────────────X────T───────────S───S───
                    │
3: ───────@────────S┼────H───────────────Z───
          │         │
4: ───────@─────────@────────────────────X───
                  └──┘
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

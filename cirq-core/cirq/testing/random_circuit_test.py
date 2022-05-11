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
import random

import numpy as np
import pytest

import cirq
import cirq.testing


def test_random_circuit_errors():
    with pytest.raises(ValueError, match='but was -1'):
        _ = cirq.testing.random_circuit(qubits=5, n_moments=5, op_density=-1)

    with pytest.raises(ValueError, match='empty'):
        _ = cirq.testing.random_circuit(qubits=5, n_moments=5, op_density=0.5, gate_domain={})

    with pytest.raises(ValueError, match='At least one'):
        _ = cirq.testing.random_circuit(qubits=0, n_moments=5, op_density=0.5)

    with pytest.raises(ValueError, match='At least one'):
        _ = cirq.testing.random_circuit(qubits=(), n_moments=5, op_density=0.5)

    with pytest.raises(
        ValueError,
        match='After removing gates that act on less than 1 qubits, gate_domain had no gates',
    ):
        _ = cirq.testing.random_circuit(
            qubits=1, n_moments=5, op_density=0.5, gate_domain={cirq.CNOT: 2}
        )


def _cases_for_random_circuit():
    i = 0
    while i < 10:
        n_qubits = random.randint(1, 20)
        n_moments = random.randint(1, 10)
        op_density = random.random()
        if random.randint(0, 1):
            gate_domain = dict(
                random.sample(
                    tuple(cirq.testing.DEFAULT_GATE_DOMAIN.items()),
                    random.randint(1, len(cirq.testing.DEFAULT_GATE_DOMAIN)),
                )
            )
            # Sometimes we generate gate domains whose gates all act on a
            # number of qubits greater that the number of qubits for the
            # circuit. In this case, try again.
            if all(n > n_qubits for n in gate_domain.values()):
                # coverage: ignore
                continue
        else:
            gate_domain = None
        pass_qubits = random.choice((True, False))
        yield (n_qubits, n_moments, op_density, gate_domain, pass_qubits)
        i += 1


@pytest.mark.parametrize(
    'n_qubits,n_moments,op_density,gate_domain,pass_qubits', _cases_for_random_circuit()
)
def test_random_circuit(
    n_qubits: Union[int, Sequence[cirq.Qid]],
    n_moments: int,
    op_density: float,
    gate_domain: Optional[Dict[cirq.Gate, int]],
    pass_qubits: bool,
):
    qubit_set = cirq.LineQubit.range(n_qubits)
    qubit_arg = qubit_set if pass_qubits else n_qubits
    circuit = cirq.testing.random_circuit(qubit_arg, n_moments, op_density, gate_domain)
    if qubit_arg is qubit_set:
        assert circuit.all_qubits().issubset(qubit_set)
    assert len(circuit) == n_moments
    if gate_domain is None:
        gate_domain = cirq.testing.DEFAULT_GATE_DOMAIN
    assert set(cast(cirq.GateOperation, op).gate for op in circuit.all_operations()).issubset(
        gate_domain
    )


@pytest.mark.parametrize('seed', [random.randint(0, 2**32) for _ in range(10)])
def test_random_circuit_reproducible_with_seed(seed):
    wrappers = (lambda s: s, np.random.RandomState)
    circuits = [
        cirq.testing.random_circuit(
            qubits=10, n_moments=10, op_density=0.7, random_state=wrapper(seed)
        )
        for wrapper in wrappers
        for _ in range(2)
    ]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*circuits)


def test_random_circuit_not_expected_number_of_qubits():

    circuit = cirq.testing.random_circuit(
        qubits=3, n_moments=1, op_density=1.0, gate_domain={cirq.CNOT: 2}
    )
    # Despite having an op density of 1, we always only end up acting on
    # two qubits.
    assert len(circuit.all_qubits()) == 2


def test_random_circuit_reproducible_between_runs():
    circuit = cirq.testing.random_circuit(5, 8, 0.5, random_state=77)
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


def test_random_two_qubit_circuit_with_czs():
    num_czs = lambda circuit: len(
        [o for o in circuit.all_operations() if isinstance(o.gate, cirq.CZPowGate)]
    )

    c = cirq.testing.random_two_qubit_circuit_with_czs()
    assert num_czs(c) == 3
    assert {cirq.NamedQubit('q0'), cirq.NamedQubit('q1')} == c.all_qubits()
    assert all(isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations)
    assert c[0].qubits == c.all_qubits()

    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=0)
    assert num_czs(c) == 0
    assert {cirq.NamedQubit('q0'), cirq.NamedQubit('q1')} == c.all_qubits()
    assert all(isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations)
    assert c[0].qubits == c.all_qubits()

    a, b = cirq.LineQubit.range(2)
    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=1, q1=b)
    assert num_czs(c) == 1
    assert {b, cirq.NamedQubit('q0')} == c.all_qubits()
    assert all(isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations)
    assert c[0].qubits == c.all_qubits()

    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=2, q0=a)
    assert num_czs(c) == 2
    assert {a, cirq.NamedQubit('q1')} == c.all_qubits()
    assert all(isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations)
    assert c[0].qubits == c.all_qubits()

    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=3, q0=a, q1=b)
    assert num_czs(c) == 3
    assert c.all_qubits() == {a, b}
    assert all(isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations)
    assert c[0].qubits == c.all_qubits()

    seed = 77

    c1 = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=4, q0=a, q1=b, random_state=seed)
    assert num_czs(c1) == 4
    assert c1.all_qubits() == {a, b}

    c2 = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=4, q0=a, q1=b, random_state=seed)

    assert c1 == c2

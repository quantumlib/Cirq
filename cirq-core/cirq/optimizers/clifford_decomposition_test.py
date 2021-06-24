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

import pytest
import numpy as np

import cirq
from cirq.testing import assert_allclose_up_to_global_phase


def test_misaligned_qubits():
    qubits = cirq.LineQubit.range(1)
    tableau = cirq.CliffordTableau(num_qubits=2)
    with pytest.raises(ValueError):
        cirq.decompose_clifford_tableau_to_operations(qubits, tableau)


def test_clifford_decompose_one_qubit():
    """Two random instance for one qubit decomposition."""
    qubits = cirq.LineQubit.range(1)
    args = cirq.ActOnCliffordTableauArgs(
        tableau=cirq.CliffordTableau(num_qubits=1),
        qubits=qubits,
        prng=np.random.RandomState(),
        log_of_measurement_results={},
    )
    cirq.act_on(cirq.X, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.S, args, qubits=[qubits[0]], allow_decompose=False)
    expect_circ = cirq.Circuit(cirq.X(qubits[0]), cirq.H(qubits[0]), cirq.S(qubits[0]))
    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-7)

    qubits = cirq.LineQubit.range(1)
    args = cirq.ActOnCliffordTableauArgs(
        tableau=cirq.CliffordTableau(num_qubits=1),
        qubits=qubits,
        prng=np.random.RandomState(),
        log_of_measurement_results={},
    )
    cirq.act_on(cirq.Z, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.S, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.X, args, qubits=[qubits[0]], allow_decompose=False)
    expect_circ = cirq.Circuit(
        cirq.Z(qubits[0]),
        cirq.H(qubits[0]),
        cirq.S(qubits[0]),
        cirq.H(qubits[0]),
        cirq.X(qubits[0]),
    )
    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-7)


def test_clifford_decompose_two_qubits():
    """Two random instance for two qubits decomposition."""
    qubits = cirq.LineQubit.range(2)
    args = cirq.ActOnCliffordTableauArgs(
        tableau=cirq.CliffordTableau(num_qubits=2),
        qubits=qubits,
        prng=np.random.RandomState(),
        log_of_measurement_results={},
    )
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CNOT, args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    expect_circ = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-7)

    qubits = cirq.LineQubit.range(2)
    args = cirq.ActOnCliffordTableauArgs(
        tableau=cirq.CliffordTableau(num_qubits=2),
        qubits=qubits,
        prng=np.random.RandomState(),
        log_of_measurement_results={},
    )
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CNOT, args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.S, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.X, args, qubits=[qubits[1]], allow_decompose=False)
    expect_circ = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.H(qubits[0]),
        cirq.S(qubits[0]),
        cirq.X(qubits[1]),
    )

    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-7)


def test_clifford_decompose_by_unitary():
    """Validate the decomposition of random Clifford Tableau by unitary matrix.

    Due to the exponential growth in dimension, it cannot validate very large number of qubits.
    """
    n, num_ops = 5, 20
    gate_candidate = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.CNOT, cirq.CZ]
    for seed in range(100):
        prng = np.random.RandomState(seed)
        t = cirq.CliffordTableau(num_qubits=n)
        qubits = cirq.LineQubit.range(n)
        expect_circ = cirq.Circuit()
        args = cirq.ActOnCliffordTableauArgs(
            tableau=t, qubits=qubits, prng=prng, log_of_measurement_results={}
        )
        for _ in range(num_ops):
            g = prng.randint(len(gate_candidate))
            indices = (prng.randint(n),) if g < 5 else prng.choice(n, 2, replace=False)
            cirq.act_on(
                gate_candidate[g], args, qubits=[qubits[i] for i in indices], allow_decompose=False
            )
            expect_circ.append(gate_candidate[g].on(*[qubits[i] for i in indices]))
        ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
        circ = cirq.Circuit(ops)
        circ.append(cirq.I.on_each(qubits))
        expect_circ.append(cirq.I.on_each(qubits))
        assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-7)


def test_clifford_decompose_by_reconstruction():
    """Validate the decomposition of random Clifford Tableau by reconstruction.

    This approach can validate large number of qubits compared with the unitary one.
    """
    n, num_ops = 100, 500
    gate_candidate = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.CNOT, cirq.CZ]
    for seed in range(10):
        prng = np.random.RandomState(seed)
        t = cirq.CliffordTableau(num_qubits=n)
        qubits = cirq.LineQubit.range(n)
        expect_circ = cirq.Circuit()
        args = cirq.ActOnCliffordTableauArgs(
            tableau=t, qubits=qubits, prng=prng, log_of_measurement_results={}
        )
        for _ in range(num_ops):
            g = prng.randint(len(gate_candidate))
            indices = (prng.randint(n),) if g < 5 else prng.choice(n, 2, replace=False)
            cirq.act_on(
                gate_candidate[g], args, qubits=[qubits[i] for i in indices], allow_decompose=False
            )
            expect_circ.append(gate_candidate[g].on(*[qubits[i] for i in indices]))
        ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)

        reconstruct_t = cirq.CliffordTableau(num_qubits=n)
        reconstruct_args = cirq.ActOnCliffordTableauArgs(
            tableau=reconstruct_t, qubits=qubits, prng=prng, log_of_measurement_results={}
        )
        for op in ops:
            cirq.act_on(op.gate, reconstruct_args, qubits=op.qubits, allow_decompose=False)

        assert t == reconstruct_t

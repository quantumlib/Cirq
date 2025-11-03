# Copyright 2025 The Cirq Developers
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

from __future__ import annotations

import numpy as np
import pytest

import cirq
from cirq import I, X, Y, Z, ZPowGate
from cirq.transformers.gauge_compiling.multi_moment_cphase_gauge import (
    _PauliAndZPow,
    CPhaseGaugeTransformerMM,
)


def test_gauge_on_single_cphase():
    """Test case.
    Input:
    0: ───@───────
          │
    1: ───@^0.2───
    Example output:
    0: ───X───@────────PhXZ(a=0,x=1,z=0)───
              │
    1: ───I───@^-0.2───Z^0.2───────────────
    """
    q0, q1 = cirq.LineQubit.range(2)

    input_circuit = cirq.Circuit(cirq.Moment(cirq.CZ(q0, q1) ** 0.2))

    class _TestCPhaseGaugeTransformerMM(CPhaseGaugeTransformerMM):
        def sample_left_moment(
            self,
            active_qubits: frozenset[cirq.Qid],
            rng: np.random.Generator = np.random.default_rng(),
        ) -> cirq.Moment:
            return cirq.Moment(g1(q0), g2(q1))

    for g1 in [X, Y, Z, I]:
        for g2 in [X, Y, Z, I]:  # Test with all possible samples of the left moment.
            cphase_transformer = _TestCPhaseGaugeTransformerMM()
            output_circuit = cphase_transformer(input_circuit)
            cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
                input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
            )


def test_gauge_on_cz_moments():
    """Test case.
    Input:
              ┌──┐
    0: ───@────@─────H───────@───@───
          │    │             │   │
    1: ───@────┼@────────────@───@───
               ││
    2: ───@────@┼────────@───@───@───
          │     │        │   │   │
    3: ───@─────@────────@───@───@───
              └──┘
    Example output:
                  ┌──┐
    0: ───X───@────@─────PhXZ(a=0,x=1,z=1)──────H───X───────@───@───PhXZ(a=0,x=1,z=2)────
              │    │                                        │   │
    1: ───I───@────┼@────Z──────────────────────────X───────@───@───PhXZ(a=2,x=1,z=-2)───
                   ││
    2: ───Y───@────@┼────PhXZ(a=1.5,x=1,z=-1)───────Z───@───@───@───Z────────────────────
              │     │                                   │   │   │
    3: ───Z───@─────@────Z^0────────────────────────I───@───@───@───Z^0──────────────────
                  └──┘
    """
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    input_circuit = cirq.Circuit(
        cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3)),
        cirq.Moment(cirq.CZ(q0, q2), cirq.CZ(q1, q3)),
        cirq.Moment(cirq.H(q0)),
        cirq.Moment(cirq.CZ(q2, q3)),
        cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3)),
        cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3)),
    )
    transformer = CPhaseGaugeTransformerMM()

    output_circuit = transformer(input_circuit)
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
    )


def test_is_target_moment():
    q0, q1, q2 = cirq.LineQubit.range(3)

    target_moments = [
        cirq.Moment(cirq.CZ(q0, q1) ** 0.2),
        cirq.Moment(cirq.CZ(q0, q1) ** 0.2, cirq.X(q2)),
    ]
    non_target_moments = [
        cirq.Moment(cirq.X(q0), cirq.Y(q1)),
        cirq.Moment(cirq.CZ(q0, q1) ** 0.2, cirq.Rz(rads=-0.8).on(q2)),
        cirq.Moment(cirq.CZ(q0, q1).with_tags("ignore")),
        cirq.Moment(cirq.CZ(q0, q1)).with_tags("ignore"),
    ]
    cphase_transformer = CPhaseGaugeTransformerMM(supported_gates=cirq.Gateset(cirq.Pauli))
    for m in target_moments:
        assert cphase_transformer.is_target_moment(m)
    for m in non_target_moments:
        assert not cphase_transformer.is_target_moment(
            m, cirq.TransformerContext(tags_to_ignore={'ignore'})
        )


def test_gauge_on_cphase_moments():
    """Test case.
    Input:
                  ┌──┐
    0: ───@────────@─────H───Rz(-0.255π)───────────@───────@───────
          │        │                               │       │
    1: ───@^0.2────┼@──────────────────────────────@^0.1───@───────
                   ││
    2: ───@────────@┼────────@─────────────@───────@───────@───────
          │         │        │             │       │       │
    3: ───@─────────@────────@^0.2─────────@^0.2───@───────@^0.2───
                  └──┘
    Example output:
                       ┌──┐
    0: ───Y───@─────────@─────PhXZ(a=0,x=1,z=0)───H───X───Rz(0.255π)────────────@───────@────────PhXZ(a=0,x=1,z=1.1)───
              │         │                                                       │       │
    1: ───I───@^-0.2────┼@────Z^0.2───────────────────Y─────────────────────────@^0.1───@────────PhXZ(a=0,x=1,z=0.1)───
                        ││
    2: ───X───@─────────@┼────PhXZ(a=0,x=1,z=1)───────X───@────────────@────────@───────@────────PhXZ(a=0,x=1,z=0)─────
              │          │                                │            │        │       │
    3: ───Z───@──────────@────I───────────────────────I───@^-0.2───────@^-0.2───@───────@^-0.2───Z^-0.4────────────────
                       └──┘
    """  # noqa: E501
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    cphase_transformer = CPhaseGaugeTransformerMM()
    for seed in range(5):
        input_circuit = cirq.Circuit(
            cirq.Moment(cirq.CZ(q0, q1) ** 0.2, cirq.CZ(q2, q3)),
            cirq.Moment(cirq.CZ(q0, q2), cirq.CZ(q1, q3)),
            cirq.Moment(cirq.H(q0)),
            cirq.Moment(cirq.CZ(q2, q3) ** 0.2, cirq.Rz(rads=-0.8).on(q0)),
            cirq.Moment(cirq.CZ(q2, q3) ** 0.2),
            cirq.Moment(cirq.CZ(q0, q1) ** 0.1, cirq.CZ(q2, q3)),
            cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3) ** 0.2),
        )

        output_circuit = cphase_transformer(input_circuit, rng_or_seed=seed)
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
            input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
        )


def test_gauge_on_czpow_only_moments():
    q0, q1, q2 = cirq.LineQubit.range(3)

    input_circuit = cirq.Circuit(cirq.Moment(cirq.CZ(q0, q1) ** 0.2, X(q2)))
    cphase_transformer = CPhaseGaugeTransformerMM(supported_gates=cirq.Gateset())
    output_circuit = cphase_transformer(input_circuit)

    # Since X isn't in supported_gates, the moment won't be gauged.
    assert input_circuit == output_circuit


def test_gauge_on_supported_gates():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    cphase_transformer = CPhaseGaugeTransformerMM()
    for g1 in [X, Z**0.6, I, Z]:
        for g2 in [Y, cirq.Rz(rads=0.2), Z**0.7]:
            input_circuit = cirq.Circuit(
                cirq.Moment(cirq.CZ(q0, q1) ** 0.2, g1(q2), g2(q3)),
                cirq.Moment(cirq.CZ(q0, q2), g2(q1), g1(q3)),
            )
            output_circuit = cphase_transformer(input_circuit)
            cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
                input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
            )


def test_gauge_on_unsupported_gates():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)

    cphase_transformer = CPhaseGaugeTransformerMM(supported_gates=cirq.Gateset(cirq.CNOT))
    with pytest.raises(ValueError, match="Gate type .* is not supported."):
        cphase_transformer(cirq.Circuit(cirq.CNOT(q0, q1), cirq.CZ(q2, q3)))


def test_pauli_and_phxz_util_str():
    assert str(_PauliAndZPow(pauli=X)) == '─X──Z**0─'
    assert str(_PauliAndZPow(pauli=X, zpow=Z**0.1)) == '─X──Z**0.1─'


def test_pauli_and_phxz_util_gate_merges():
    """Tests _PauliAndZPow's merge_left() and merge_right()."""
    for left_pauli in [X, Y, Z, I]:
        for right_pauli in [X, Y, Z, I]:
            left = _PauliAndZPow(pauli=left_pauli, zpow=ZPowGate(exponent=0.2))
            right = _PauliAndZPow(pauli=right_pauli, zpow=ZPowGate(exponent=0.6))
            merge1 = right.merge_left(left)
            merge2 = left.merge_right(right)

            assert np.allclose(
                cirq.unitary(merge1.to_single_qubit_gate()),
                cirq.unitary(merge2.to_single_qubit_gate()),
            )
            q = cirq.LineQubit(0)
            cirq.testing.assert_allclose_up_to_global_phase(
                cirq.unitary(
                    cirq.Circuit(
                        left.to_single_qubit_gate().on(q), right.to_single_qubit_gate().on(q)
                    )
                ),
                cirq.unitary(merge1.to_single_qubit_gate()),
                atol=1e-6,
            )


def test_pauli_and_phxz_util_to_1q_gate():
    """Tests _PauliAndZPow.to_single_qubit_gate()."""
    q = cirq.LineQubit(0)
    for pauli in [cirq.X, cirq.Y, cirq.Z, cirq.I]:
        for zpow in [cirq.ZPowGate(exponent=exp) for exp in [0, 0.1, 0.5, 1, 10.2]]:
            cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
                cirq.Circuit(pauli(q), zpow(q)),
                cirq.Circuit(_PauliAndZPow(pauli=pauli, zpow=zpow).to_single_qubit_gate().on(q)),
                {q: q},
            )


def test_deep_not_supported():
    with pytest.raises(ValueError, match="GaugeTransformer cannot be used with deep=True"):
        t = CPhaseGaugeTransformerMM()
        t(cirq.Circuit(), context=cirq.TransformerContext(deep=True))

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

from copy import deepcopy

import numpy as np

import cirq
from cirq.transformers.gauge_compiling.cphase_gauge import (
    _PhasedXYAndRz,
    CPhaseGaugeTransformer,
    CPhaseGaugeTransformerMM,
)
from cirq.transformers.gauge_compiling.gauge_compiling_test_utils import GaugeTester


class TestCPhaseGauge_0_3(GaugeTester):
    two_qubit_gate = cirq.CZ**0.3
    gauge_transformer = CPhaseGaugeTransformer


class TestCPhaseGauge_m0_3(GaugeTester):
    two_qubit_gate = cirq.CZ ** (-0.3)
    gauge_transformer = CPhaseGaugeTransformer


class TestCPhaseGauge_0_1(GaugeTester):
    two_qubit_gate = cirq.CZ**0.1
    gauge_transformer = CPhaseGaugeTransformer


class TestCPhaseGauge_m0_1(GaugeTester):
    two_qubit_gate = cirq.CZ ** (-0.1)
    gauge_transformer = CPhaseGaugeTransformer


class TestCPhaseGauge_0_7(GaugeTester):
    two_qubit_gate = cirq.CZ**0.7
    gauge_transformer = CPhaseGaugeTransformer


def test_single_cphase():
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
    transformer = CPhaseGaugeTransformerMM

    output_circuit = transformer(input_circuit, prng=np.random.default_rng())

    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
    )


def test_multi_layer_pull_through_all_czs():
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
    transformer = CPhaseGaugeTransformerMM

    output_circuit = transformer(input_circuit, prng=np.random.default_rng())
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
    )


def test_multi_layer_pull_through():
    """Test case.
    Input:
                  ┌──┐
    0: ───@────────@─────H───Rz(-0.255π)───@───────@───────
          │        │                       │       │
    1: ───@^0.2────┼@──────────────────────@^0.1───@───────
                   ││
    2: ───@────────@┼────────@─────────────@───────@───────
          │         │        │             │       │
    3: ───@─────────@────────@^0.2─────────@───────@^0.2───
                  └──┘
    Example output:
                       ┌──┐
    0: ───Z───@─────────@─────Z^0.2────────────────H───I───────────@────────@───────Z^0.845──────────────────  # pylint: disable=line-too-long
              │         │                                          │        │
    1: ───X───@^-0.2────┼@────PhXZ(a=0,x=1,z=1)────────X───────────@^-0.1───@───────PhXZ(a=0,x=1,z=0)────────  # pylint: disable=line-too-long
                        ││
    2: ───X───@─────────@┼────PhXZ(a=0,x=1,z=1)────────Y───@───────@────────@───────PhXZ(a=0.5,x=1,z=1.4)────  # pylint: disable=line-too-long
              │          │                                 │       │        │
    3: ───X───@──────────@────PhXZ(a=2,x=1,z=-2)───────Y───@^0.2───@────────@^0.2───PhXZ(a=1.9,x=1,z=-1.4)───  # pylint: disable=line-too-long
                       └──┘
    """
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    for _ in range(10):
        input_circuit = cirq.Circuit(
            cirq.Moment(cirq.CZ(q0, q1) ** 0.2, cirq.CZ(q2, q3)),
            cirq.Moment(cirq.CZ(q0, q2), cirq.CZ(q1, q3)),
            cirq.Moment(cirq.H(q0)),
            cirq.Moment(cirq.CZ(q2, q3) ** 0.2, cirq.Rz(rads=-0.8).on(q0)),
            cirq.Moment(cirq.CZ(q0, q1) ** 0.1, cirq.CZ(q2, q3)),
            cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3) ** 0.2),
        )
        transformer = CPhaseGaugeTransformerMM

        output_circuit = transformer(input_circuit, prng=np.random.default_rng())
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
            input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
        )


def test_phxyrz_str():
    phxyrz = _PhasedXYAndRz(pauli=cirq.X, phase_exp=0.1)
    assert str(phxyrz) == "─X──Rz(0)──phase(e^{i0.1π})─"


def test_phxyrz_to_single_gate():
    q = cirq.LineQubit(0)
    for pauli in [cirq.X, cirq.Y, cirq.Z, cirq.I]:
        for rz_rads in [0.0, np.random.uniform(0, 2 * np.pi)]:
            for phase_exp in [0.0, np.random.uniform(0, 1)]:
                phxyrz = _PhasedXYAndRz(pauli=pauli, rz_rads=rz_rads, phase_exp=phase_exp)
                assert np.allclose(
                    cirq.unitary(phxyrz.to_single_gate()),
                    cirq.unitary(
                        cirq.Circuit(
                            pauli(q),
                            cirq.Rz(rads=rz_rads).on(q),
                            cirq.global_phase_operation(coefficient=np.exp(1j * phase_exp * np.pi)),
                        )
                    ),
                )


def test_phxyrz_merge():
    for left_pauli in [cirq.X, cirq.Y, cirq.Z, cirq.I]:
        for right_pauli in [cirq.X, cirq.Y, cirq.Z, cirq.I]:
            left = _PhasedXYAndRz(pauli=left_pauli, rz_rads=0.1, phase_exp=0.2)
            right = _PhasedXYAndRz(pauli=right_pauli, rz_rads=0.3, phase_exp=0.4)
            merge1 = deepcopy(right)
            merge1.merge_left(left)
            merge2 = deepcopy(left)
            merge2.merge_right(right)

            assert merge1 == merge2
            q = cirq.LineQubit(0)
            assert np.allclose(
                cirq.unitary(
                    cirq.Circuit(left.to_single_gate().on(q), right.to_single_gate().on(q))
                ),
                cirq.unitary(merge1.to_single_gate()),
                atol=1e-10,
            )

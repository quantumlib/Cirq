# Copyright 2024 The Cirq Developers
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
from cirq.transformers.gauge_compiling import SqrtCZGaugeTransformer
from cirq.transformers.gauge_compiling.gauge_compiling_test_utils import GaugeTester


class TestSqrtCZGauge(GaugeTester):
    two_qubit_gate = cirq.CZ**0.5
    gauge_transformer = SqrtCZGaugeTransformer


class TestAdjointSqrtCZGauge(GaugeTester):
    two_qubit_gate = cirq.CZ**-0.5
    gauge_transformer = SqrtCZGaugeTransformer


# Gates that the transformer's target gateset accepts because they equal CZ**0.5 or CZ**-0.5 up
# to global phase, although most of them are not equal to CZ**0.5 or CZ**-0.5 under `==`.
# `ControlledGate(S)` is also what the OpenQASM importer produces for `cp(pi/2)`.
_SQRT_CZ_LIKE_GATES = (
    # equivalent to _SQRT_CZ
    cirq.CZ**0.5,
    cirq.ControlledGate(cirq.S),
    cirq.CZPowGate(exponent=0.5, global_shift=0.25),
    cirq.MatrixGate(np.diag([1, 1, 1, 1j])),
    # equivalent to _ADJ_SQRT_CZ
    cirq.CZ**-0.5,
    cirq.ControlledGate(cirq.S**-1),
    cirq.CZPowGate(exponent=-0.5, global_shift=0.25),
    cirq.MatrixGate(np.diag([1, 1, 1, -1j])),
)


@pytest.mark.parametrize('gate', _SQRT_CZ_LIKE_GATES)
def test_transformer_on_equivalent_representations(gate: cirq.Gate) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(20 * [cirq.H(q0), gate(q0, q1), cirq.H(q1)])
    prng = cirq.value.parse_random_generator(None)
    nc = SqrtCZGaugeTransformer(c, prng=prng)
    # confirm some of the gate appearances were transformed by SqrtCZGauge.sample
    assert any(op.gate not in (cirq.H, gate) for op in nc.all_operations())
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        nc, c, qubit_map={q0: q0, q1: q1}
    )


@pytest.mark.parametrize('gate', _SQRT_CZ_LIKE_GATES)
def test_as_sweep_on_equivalent_representations(gate: cirq.Gate) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(20 * [cirq.H(q0), gate(q0, q1), cirq.H(q1)])
    prng = cirq.value.parse_random_generator(None)
    parameterized_circuit, sweep = SqrtCZGaugeTransformer.as_sweep(c, N=2, prng=prng)
    assert cirq.is_parameterized(parameterized_circuit)
    # check the parameterized circuit for each set of parameters
    for params in sweep:
        compiled_circuit = cirq.resolve_parameters(parameterized_circuit, params)
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
            compiled_circuit, c, qubit_map={q0: q0, q1: q1}
        )

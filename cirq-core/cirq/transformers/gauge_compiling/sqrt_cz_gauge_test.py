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

from unittest import mock

import numpy as np
import pytest
import sympy

import cirq
from cirq.transformers.gauge_compiling import SqrtCZGaugeTransformer
from cirq.transformers.gauge_compiling.gauge_compiling_test_utils import GaugeTester
from cirq.transformers.gauge_compiling.sqrt_cz_gauge import _symbolize_as_cz_pow, SqrtCZGauge


class TestSqrtCZGauge(GaugeTester):
    two_qubit_gate = cirq.CZ**0.5
    gauge_transformer = SqrtCZGaugeTransformer


class TestAdjointSqrtCZGauge(GaugeTester):
    two_qubit_gate = cirq.CZ**-0.5
    gauge_transformer = SqrtCZGaugeTransformer


class _UnitaryOnlyGate(cirq.Gate):
    """A two-qubit diagonal gate that implements `_unitary_` but not `__pow__` or `_decompose_`."""

    def __init__(self, diagonal: tuple[complex, ...]) -> None:
        self._diagonal = diagonal

    def _num_qubits_(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return np.diag(self._diagonal)

    def __repr__(self) -> str:
        return f'_UnitaryOnlyGate({self._diagonal!r})'


# Gates that the transformer's target gateset accepts because they equal CZ**0.5 or CZ**-0.5 up
# to global phase, although most of them are not equal to CZ**0.5 or CZ**-0.5 under `==`.
# `ControlledGate(S)` is also what the OpenQASM importer produces for `cp(pi/2)`.
_SQRT_CZ_REPRESENTATIONS = [
    cirq.CZ**0.5,
    cirq.CZ**-1.5,
    cirq.ControlledGate(cirq.S),
    cirq.CZPowGate(exponent=0.5, global_shift=0.25),
    cirq.MatrixGate(np.diag([1, 1, 1, 1j])),
    _UnitaryOnlyGate(tuple(np.exp(0.3j) * np.array([1, 1, 1, 1j]))),
]
_ADJ_SQRT_CZ_REPRESENTATIONS = [
    cirq.CZ**-0.5,
    cirq.CZ**1.5,
    cirq.ControlledGate(cirq.S**-1),
    cirq.CZPowGate(exponent=-0.5, global_shift=0.25),
    cirq.MatrixGate(np.diag([1, 1, 1, -1j])),
    _UnitaryOnlyGate(tuple(np.exp(0.3j) * np.array([1, 1, 1, -1j]))),
]
# Two-qubit gates that equal neither CZ**0.5 nor CZ**-0.5 up to global phase.
_UNSUPPORTED_GATES = [
    cirq.CZ,
    cirq.CZ**0.25,
    cirq.CNOT,
    cirq.MatrixGate(np.diag([1, 1, 1, -1])),
    _UnitaryOnlyGate((1, 1, 1, np.exp(0.25j * np.pi))),
]


def _gauge_as_circuit(gauge, q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    return cirq.Circuit(
        [g(q0) for g in gauge.pre_q0],
        [g(q1) for g in gauge.pre_q1],
        gauge.on(q0, q1),
        [g(q0) for g in gauge.post_q0],
        [g(q1) for g in gauge.post_q1],
    )


@pytest.mark.parametrize('gate', _SQRT_CZ_REPRESENTATIONS + _ADJ_SQRT_CZ_REPRESENTATIONS)
@pytest.mark.parametrize('swap_qubits', [False, True])
def test_non_identity_gauges_preserve_unitary(gate, swap_qubits) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    prng = mock.create_autospec(np.random.Generator, instance=True)
    prng.choice.side_effect = [False, swap_qubits]
    gauge = SqrtCZGauge().sample(gate, prng)
    assert gauge.swap_qubits == swap_qubits
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(_gauge_as_circuit(gauge, q0, q1)), cirq.unitary(gate(q0, q1)), atol=1e-8
    )


@pytest.mark.parametrize('gate', _SQRT_CZ_REPRESENTATIONS + _ADJ_SQRT_CZ_REPRESENTATIONS)
@pytest.mark.parametrize('seed', range(4))
def test_transformer_on_equivalent_representations(gate, seed) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q0), gate(q0, q1), cirq.H(q1))
    nc = SqrtCZGaugeTransformer(c, prng=np.random.default_rng(seed))
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        nc, c, qubit_map={q: q for q in c.all_qubits()}
    )


@pytest.mark.parametrize('gate', _SQRT_CZ_REPRESENTATIONS + _ADJ_SQRT_CZ_REPRESENTATIONS)
def test_as_sweep_on_equivalent_representations(gate) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q0), gate(q0, q1), cirq.H(q1))
    parameterized_circuit, sweep = SqrtCZGaugeTransformer.as_sweep(
        c, N=16, prng=np.random.default_rng(0)
    )
    for params in sweep:
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary(cirq.resolve_parameters(parameterized_circuit, params)),
            cirq.unitary(c),
            atol=1e-8,
        )


@pytest.mark.parametrize('gate', _UNSUPPORTED_GATES)
@pytest.mark.parametrize('choices', [[True], [False, False], [False, True]])
def test_sample_rejects_unsupported_gate(gate, choices) -> None:
    prng = mock.create_autospec(np.random.Generator, instance=True)
    prng.choice.side_effect = choices
    with pytest.raises(ValueError, match=r'CZ\*\*0\.5 or CZ\*\*-0\.5 up to global phase'):
        SqrtCZGauge().sample(gate, prng)


@pytest.mark.parametrize(
    'gate', [g for g in _UNSUPPORTED_GATES if not isinstance(g, cirq.CZPowGate)]
)
def test_symbolize_rejects_unsupported_gate(gate) -> None:
    with pytest.raises(ValueError, match="Can't symbolize non-CZPowGate"):
        _symbolize_as_cz_pow(gate, [sympy.Symbol('s')])

# Copyright 2026 The Cirq Developers
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

import cirq
from cirq import value
from cirq.testing import random_special_unitary
from cirq.transformers.heuristic_decompositions.two_qubit_numerical_optimization import (
    two_qubit_gate_numerical_compilation,
    TwoQubitNumericalCompiler,
)

_CZ = cirq.unitary(cirq.CZ)
_ISWAP = cirq.unitary(cirq.ISWAP)
_SQRT_ISWAP = cirq.unitary(cirq.SQRT_ISWAP)
_SYCAMORE = cirq.unitary(cirq.FSimGate(np.pi / 2, np.pi / 6))

_BASE_GATES = {'CZ': _CZ, 'iSWAP': _ISWAP, 'Sycamore': _SYCAMORE}

# All seeds below are fixed per test (rather than drawn from a shared RNG) so
# that the tests stay deterministic under pytest-randomly shuffling. They were
# selected to give the numerically expected outcomes; see each test.


def _reconstruct_actual_gate(result) -> np.ndarray:
    """Rebuild actual_gate from local_unitaries and base_gate_unitary."""
    actual = np.eye(4, dtype=complex)
    for i, (k0, k1) in enumerate(result.local_unitaries):
        actual = np.kron(k0, k1) @ actual
        if i < result.num_base_gates:
            actual = result.base_gate_unitary @ actual
    return actual


def _compile(*args, **kwargs):
    """Compile with 3 restarts instead of the default 5, to keep tests fast.

    Exercises the same code paths; the per-test seeds were selected with
    this value. Explicitly passed num_restarts is left untouched.
    """
    kwargs.setdefault('num_restarts', 3)
    return two_qubit_gate_numerical_compilation(*args, **kwargs)


@pytest.mark.parametrize('base_gate_name', list(_BASE_GATES))
@pytest.mark.parametrize('seed', [1, 2, 3])
def test_exact_compilation_random_su4(base_gate_name: str, seed: int) -> None:
    """Random SU(4) targets compile exactly (Fd >= 1 - 1e-8) with 3 base gates.

    Matches the paper's finding (arXiv:2106.15490, Sec. VII.A) for NuOp-100%
    (exact decomposition): NuOp uses 3 CZ, 3 SYC or 3 sqrt-iSWAP gates for
    random unitaries.
    """
    target = random_special_unitary(4, random_state=value.parse_random_state(seed))
    result = _compile(target, _BASE_GATES[base_gate_name], random_state=seed)
    assert result.success
    assert result.num_base_gates == 3
    assert result.decomposition_fidelity >= 1 - 1e-6


@pytest.mark.parametrize('base_gate_name', list(_BASE_GATES))
def test_compilation_of_base_gate_itself(base_gate_name: str) -> None:
    """Compiling the base gate itself needs exactly one layer."""
    base_gate = _BASE_GATES[base_gate_name]
    result = _compile(base_gate, base_gate, random_state=10)
    assert result.success
    assert result.num_base_gates == 1
    assert len(result.local_unitaries) == 2
    assert result.decomposition_fidelity >= 1 - 1e-8
    cirq.testing.assert_allclose_up_to_global_phase(result.actual_gate, base_gate, atol=1e-6)


def test_compilation_of_locally_equivalent_target() -> None:
    """A target locally equivalent to the base gate needs exactly one layer."""
    target = (
        np.kron(cirq.unitary(cirq.H), cirq.unitary(cirq.S))
        @ _CZ
        @ np.kron(np.eye(2), cirq.unitary(cirq.H))
    )
    result = _compile(target, _CZ, random_state=11)
    assert result.success
    assert result.num_base_gates == 1


def test_compilation_of_zz_interaction() -> None:
    """QAOA-style ZZ interactions compile to 2 CZ gates (paper, Fig. 2)."""
    target = cirq.unitary(cirq.ZZPowGate(exponent=0.3))
    result = _compile(target, _CZ, random_state=12)
    assert result.success
    assert result.num_base_gates == 2
    assert result.decomposition_fidelity >= 1 - 1e-8


def test_approximate_compilation_uses_fewer_gates() -> None:
    """A relaxed fidelity threshold yields decompositions with fewer base gates."""
    # Target seed chosen so that the best 2-CZ decomposition has Fd in
    # [0.99, 1 - 1e-8): exact compilation needs 3 CZ gates, but the Fd >= 0.99
    # threshold is already met with 2.
    target = random_special_unitary(4, random_state=value.parse_random_state(127))
    exact = _compile(target, _CZ, random_state=13)
    approximate = _compile(target, _CZ, target_fidelity=0.99, random_state=13)
    assert exact.num_base_gates == 3
    assert approximate.success
    assert approximate.num_base_gates < exact.num_base_gates
    assert approximate.decomposition_fidelity >= 0.99


def test_noise_adaptive_compilation_prefers_higher_overall_fidelity() -> None:
    """Reproduces the noise-adaptive scenario of the paper (arXiv:2106.15490, Fig. 5).

    With CZ at 94% fidelity and sqrt-iSWAP at 70% fidelity, the compiler
    chooses an approximate 2-CZ decomposition (Fu = Fd * 0.94^2) over the exact
    3-CZ decomposition (Fu = 0.94^3) and over any sqrt-iSWAP decomposition.
    With error rates given, `success` is measured by Fu.
    """
    target = random_special_unitary(4, random_state=value.parse_random_state(122))
    result = _compile(
        target,
        [_CZ, _SQRT_ISWAP],
        base_gate_error_rates=[0.06, 0.30],
        target_fidelity=0.85,
        max_layers=3,
        random_state=14,
    )
    assert np.array_equal(result.base_gate_unitary, _CZ)
    assert result.base_gate_index == 0
    assert result.num_base_gates == 2
    assert result.hardware_fidelity is not None
    assert result.hardware_fidelity == pytest.approx(0.94**2)
    overall = result.decomposition_fidelity * result.hardware_fidelity
    assert overall > 0.94**3  # Beats the exact 3-CZ decomposition.
    assert result.success  # Fu > 0.85, even though Fd < 1 - 1e-8.


def test_noise_adaptive_compilation_prefers_more_reliable_gates() -> None:
    """More applications of a reliable base gate can beat fewer of a noisy one.

    Target: sqrt-iSWAP, decomposable exactly with either 1 sqrt-iSWAP or
    2 CZ layers. With CZ at 1% error and sqrt-iSWAP at 30% error, the 2-CZ
    decomposition (Fu = 0.99^2) wins over the 1-sqrt-iSWAP one (Fu = 0.7).
    With error rates given, `success` is measured by Fu: despite the exact
    Fd, Fu = 0.99^2 does not meet a 0.99 overall bar.
    """
    result = _compile(
        _SQRT_ISWAP,
        [_CZ, _SQRT_ISWAP],
        base_gate_error_rates=[0.01, 0.30],
        target_fidelity=0.99,
        random_state=21,
    )
    assert np.array_equal(result.base_gate_unitary, _CZ)
    assert result.base_gate_index == 0
    assert result.num_base_gates == 2
    assert result.decomposition_fidelity >= 1 - 1e-8
    assert result.hardware_fidelity is not None
    assert result.hardware_fidelity == pytest.approx(0.99**2)
    overall = result.decomposition_fidelity * result.hardware_fidelity
    assert overall > 0.7  # Beats the exact 1-sqrt-iSWAP decomposition.
    assert not result.success  # Fu = 0.99^2 < 0.99, despite the exact Fd.


def test_noise_adaptive_compilation_single_qubit_error_rates() -> None:
    target = random_special_unitary(4, random_state=value.parse_random_state(5))
    result = _compile(
        target, _CZ, base_gate_error_rates=[0.06], single_qubit_error_rates=0.001, random_state=15
    )
    num_1q_gates = 2 * (result.num_base_gates + 1)
    expected_fh = 0.94**result.num_base_gates * 0.999**num_1q_gates
    assert result.hardware_fidelity == pytest.approx(expected_fh)


def test_noise_adaptive_compilation_per_qubit_error_rates() -> None:
    """A (q0_rate, q1_rate) pair folds per-qubit calibration data into Fh."""
    target = random_special_unitary(4, random_state=value.parse_random_state(5))
    result = _compile(
        target,
        _CZ,
        base_gate_error_rates=[0.06],
        single_qubit_error_rates=(0.01, 0.02),
        random_state=15,
    )
    n = result.num_base_gates
    expected_fh = 0.94**n * (0.99 * 0.98) ** (n + 1)
    assert result.hardware_fidelity == pytest.approx(expected_fh)


def test_local_unitaries_reconstruct_actual_gate() -> None:
    target = random_special_unitary(4, random_state=value.parse_random_state(77))
    result = _compile(target, _CZ, num_restarts=1, random_state=16)
    assert len(result.local_unitaries) == result.num_base_gates + 1
    for k0, k1 in result.local_unitaries:
        assert np.allclose(k0 @ k0.conj().T, np.eye(2), atol=1e-10)
        assert np.allclose(k1 @ k1.conj().T, np.eye(2), atol=1e-10)
    assert np.allclose(_reconstruct_actual_gate(result), result.actual_gate)


def test_actual_gate_matches_target_up_to_global_phase() -> None:
    target = random_special_unitary(4, random_state=value.parse_random_state(78))
    result = _compile(target, _CZ, random_state=17)
    cirq.testing.assert_allclose_up_to_global_phase(result.actual_gate, target, atol=1e-6)


def test_determinism_with_seed() -> None:
    target = random_special_unitary(4, random_state=value.parse_random_state(79))
    result1 = _compile(target, _CZ, num_restarts=1, random_state=42)
    result2 = _compile(target, _CZ, num_restarts=1, random_state=42)
    assert np.array_equal(result1.actual_gate, result2.actual_gate)
    assert result1.num_base_gates == result2.num_base_gates


def test_max_layers_failure_mode() -> None:
    """If max_layers is too small, return the best-effort result with success=False."""
    target = random_special_unitary(4, random_state=value.parse_random_state(80))
    result = _compile(target, _CZ, max_layers=1, num_restarts=2, random_state=18)
    assert not result.success
    assert result.num_base_gates == 1
    assert result.decomposition_fidelity < 1


def test_multiple_base_gates_first_meeting_threshold_wins() -> None:
    """With several base gates, the fewest-layer decomposition is returned.

    Gate types are tried in input order per layer count, so for a target that
    needs the same number of layers of either gate, the first gate wins and
    `base_gate_index` identifies it.
    """
    target = cirq.unitary(cirq.ZZPowGate(exponent=0.3))
    result = _compile(target, [_CZ, _ISWAP], random_state=19)
    assert result.success
    assert result.num_base_gates <= 2
    assert result.base_gate_index == 0
    assert np.array_equal(result.base_gate_unitary, _CZ)


def test_numerical_compiler_wrapper() -> None:
    compiler = TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=42, num_restarts=3)
    target = random_special_unitary(4, random_state=value.parse_random_state(81))
    result = compiler.compile_two_qubit_gate(target)
    assert result.success
    assert np.array_equal(result.base_gate_unitary, _CZ)
    assert result.base_gate_index == 0


def test_numerical_compiler_equality() -> None:
    # A non-empty memoization cache does not affect equality.
    cached_compiler = TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=3)
    cached_compiler.compile_two_qubit_gate(
        random_special_unitary(4, random_state=value.parse_random_state(83))
    )
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=3),
        TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=3),
        cached_compiler,
    )
    eq.add_equality_group(TwoQubitNumericalCompiler(base_gates=(_ISWAP,), random_state=3))
    eq.add_equality_group(TwoQubitNumericalCompiler(base_gates=(_CZ, _ISWAP), random_state=3))
    eq.add_equality_group(TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=4))
    eq.add_equality_group(
        TwoQubitNumericalCompiler(base_gates=(_CZ,), max_layers=2, random_state=3)
    )
    eq.add_equality_group(
        TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=3, max_cache_size=64)
    )


def test_numerical_compiler_repr() -> None:
    cirq.testing.assert_equivalent_repr(
        TwoQubitNumericalCompiler(
            base_gates=(_CZ, _SQRT_ISWAP), base_gate_error_rates=(0.01, 0.05), random_state=5
        )
    )
    cirq.testing.assert_equivalent_repr(
        TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=5)
    )


def test_numerical_compiler_json_roundtrip() -> None:
    compiler = TwoQubitNumericalCompiler(
        base_gates=(_CZ, _SQRT_ISWAP), base_gate_error_rates=(0.01, 0.05), random_state=5
    )
    cirq.testing.assert_json_roundtrip_works(compiler)
    compiler_no_rates = TwoQubitNumericalCompiler(base_gates=(_CZ,))
    cirq.testing.assert_json_roundtrip_works(compiler_no_rates)
    compiler_pair = TwoQubitNumericalCompiler(
        base_gates=(_CZ,), single_qubit_error_rates=(0.01, 0.02), random_state=5
    )
    cirq.testing.assert_json_roundtrip_works(compiler_pair)


def test_numerical_compiler_json_rejects_live_rng() -> None:
    compiler = TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=np.random.RandomState(5))
    with pytest.raises(ValueError, match='None or an integer seed'):
        cirq.to_json(compiler)


def test_numerical_compiler_memoizes_results() -> None:
    """Identical unitaries are compiled once; distinct ones are compiled separately."""
    compiler = TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=6, num_restarts=1)
    target1 = random_special_unitary(4, random_state=value.parse_random_state(101))
    target2 = random_special_unitary(4, random_state=value.parse_random_state(102))
    with mock.patch(
        'cirq.transformers.heuristic_decompositions.two_qubit_numerical_optimization'
        '.two_qubit_gate_numerical_compilation',
        wraps=two_qubit_gate_numerical_compilation,
    ) as wrapped:
        result1 = compiler.compile_two_qubit_gate(target1)
        assert compiler.compile_two_qubit_gate(target1) is result1
        compiler.compile_two_qubit_gate(target2)
        assert wrapped.call_count == 2


def test_numerical_compiler_cache_tolerates_numerical_noise() -> None:
    """Unitaries differing only by floating-point noise share a cache entry."""
    compiler = TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=8, num_restarts=1)
    target = random_special_unitary(4, random_state=value.parse_random_state(103))
    other = random_special_unitary(4, random_state=value.parse_random_state(104))
    # Mathematically the same unitary, recomputed in a different order.
    noisy = (target @ other) @ other.conj().T
    assert noisy.tobytes() != target.tobytes()
    with mock.patch(
        'cirq.transformers.heuristic_decompositions.two_qubit_numerical_optimization'
        '.two_qubit_gate_numerical_compilation',
        wraps=two_qubit_gate_numerical_compilation,
    ) as wrapped:
        result = compiler.compile_two_qubit_gate(target)
        assert compiler.compile_two_qubit_gate(noisy) is result
        assert wrapped.call_count == 1


def test_numerical_compiler_cache_tolerates_signed_zero_noise() -> None:
    """Tiny perturbations of opposite sign at an exact-zero entry share a cache entry."""
    compiler = TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=9, num_restarts=3)
    gate1 = _CZ.copy()
    gate1[0, 1] = 1e-12
    gate2 = _CZ.copy()
    gate2[0, 1] = -1e-12
    result = compiler.compile_two_qubit_gate(gate1)
    assert compiler.compile_two_qubit_gate(gate2) is result


def test_numerical_compiler_cache_evicts_least_recently_used() -> None:
    """When the cache exceeds max_cache_size, the oldest entry is evicted."""
    compiler = TwoQubitNumericalCompiler(base_gates=(_CZ,), random_state=7, max_cache_size=2)
    targets = [np.full((4, 4), fill, dtype=complex) for fill in (1, 2, 3)]
    with mock.patch(
        'cirq.transformers.heuristic_decompositions.two_qubit_numerical_optimization'
        '.two_qubit_gate_numerical_compilation'
    ) as fake_compile:
        for target in targets:
            compiler.compile_two_qubit_gate(target)
        assert fake_compile.call_count == 3
        compiler.compile_two_qubit_gate(targets[1])  # Still cached: no new call.
        assert fake_compile.call_count == 3
        compiler.compile_two_qubit_gate(targets[0])  # Evicted: recompiled.
        assert fake_compile.call_count == 4


def test_numerical_compiler_invalid_max_cache_size() -> None:
    with pytest.raises(ValueError, match='max_cache_size must be at least 1'):
        TwoQubitNumericalCompiler(base_gates=(_CZ,), max_cache_size=0)


def test_input_validation() -> None:
    target = random_special_unitary(4, random_state=value.parse_random_state(82))
    with pytest.raises(ValueError, match='target_unitary must have shape'):
        _compile(np.eye(2), _CZ)
    with pytest.raises(ValueError, match='target_fidelity must be in'):
        _compile(target, _CZ, target_fidelity=1.5)
    with pytest.raises(ValueError, match='max_layers must be at least 1'):
        _compile(target, _CZ, max_layers=0)
    with pytest.raises(ValueError, match='num_restarts must be at least 1'):
        _compile(target, _CZ, num_restarts=0)
    with pytest.raises(ValueError, match='maxiter must be at least 1'):
        _compile(target, _CZ, maxiter=0)
    with pytest.raises(ValueError, match='base_gates must be'):
        _compile(target, np.empty((0, 4, 4)))
    with pytest.raises(ValueError, match='one error rate per base gate'):
        _compile(target, [_CZ, _ISWAP], base_gate_error_rates=[0.06])
    with pytest.raises(ValueError, match='base_gate_error_rates must be in'):
        _compile(target, _CZ, base_gate_error_rates=[np.nan])
    with pytest.raises(ValueError, match='base_gate_error_rates must be in'):
        _compile(target, _CZ, base_gate_error_rates=[1.5])
    with pytest.raises(ValueError, match='base_gate_error_rates must be in'):
        _compile(target, _CZ, base_gate_error_rates=[-0.1])
    with pytest.raises(ValueError, match='single_qubit_error_rates must be in'):
        _compile(target, _CZ, base_gate_error_rates=[0.06], single_qubit_error_rates=np.nan)
    with pytest.raises(ValueError, match='single_qubit_error_rates must be in'):
        _compile(target, _CZ, single_qubit_error_rates=2.0)
    with pytest.raises(ValueError, match='single rate or a pair'):
        _compile(target, _CZ, single_qubit_error_rates=(0.01, 0.02, 0.03))
    with pytest.raises(ValueError, match='single_qubit_error_rates must be in'):
        _compile(target, _CZ, single_qubit_error_rates=(0.01, 1.5))


def test_non_unitary_inputs_raise_value_error() -> None:
    """Non-finite or non-unitary inputs are rejected before optimization runs."""
    target = random_special_unitary(4, random_state=value.parse_random_state(20))
    with pytest.raises(ValueError, match='target_unitary must contain only finite'):
        _compile(np.full((4, 4), np.nan), _CZ)
    with pytest.raises(ValueError, match='target_unitary must be unitary'):
        _compile(2 * np.eye(4), _CZ)
    with pytest.raises(ValueError, match='base_gates must contain only finite unitaries'):
        _compile(target, np.full((4, 4), np.nan))
    with pytest.raises(ValueError, match='base_gates must contain only finite unitaries'):
        _compile(target, 2 * np.eye(4))

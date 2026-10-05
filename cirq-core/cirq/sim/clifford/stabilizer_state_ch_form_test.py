# Copyright 2020 The Cirq Developers
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

import subprocess
import sys

import numpy as np
import pytest

import cirq

# TODO: This and Clifford tableau need tests.
# GitHub issue: https://github.com/quantumlib/Cirq/issues/3021


def test_initial_state() -> None:
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.StabilizerStateChForm(initial_state=-31, num_qubits=5)
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.StabilizerStateChForm(initial_state=32, num_qubits=5)
    state = cirq.StabilizerStateChForm(initial_state=23, num_qubits=5)
    expected_state_vector = np.zeros(32)
    expected_state_vector[23] = 1
    np.testing.assert_allclose(state.state_vector(), expected_state_vector)


def test_run() -> None:
    q0, q1, q2 = (cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2))

    """
    0: ───H───@───────────────X───M───────────
              │
    1: ───────X───@───────X───────────X───M───
                  │                   │
    2: ───────────X───M───────────────@───────

    After the third moment, before the measurement, the state is |000> + |111>.
    After measurement of q2, q0 and q1 both get a bit flip, so the q0
    measurement always yields opposite of the q2 measurement. q1 has an
    additional controlled not from q2, making it yield 1 always when measured.
    If there were no measurements in the circuit, the final state would be
    |110> + |011>.
    """
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q1, q2),
        cirq.measure(q2),
        cirq.X(q1),
        cirq.X(q0),
        cirq.measure(q0),
        cirq.CNOT(q2, q1),
        cirq.measure(q1),
        strategy=cirq.InsertStrategy.NEW,
    )
    for _ in range(10):
        state = cirq.StabilizerStateChForm(num_qubits=3)
        classical_data = cirq.ClassicalDataDictionaryStore()
        for op in circuit.all_operations():
            args = cirq.StabilizerChFormSimulationState(
                qubits=list(circuit.all_qubits()),
                prng=np.random.RandomState(),
                classical_data=classical_data,
                initial_state=state,
            )
            cirq.act_on(op, args)
        measurements = {str(k): list(v[-1]) for k, v in classical_data.records.items()}
        assert measurements['q(1)'] == [1]
        assert measurements['q(0)'] != measurements['q(2)']


@pytest.mark.parametrize('n', [0, 1, 2, 3, 5, 7])
def test_to_state_vector_small_states(n: int) -> None:
    # Test initial states for sizes below the Numba threshold (n < 8)
    for init in [0, min(1, 2**n - 1), 2**n - 1 if n > 0 else 0]:
        state = cirq.StabilizerStateChForm(n, initial_state=init)
        fallback = state._to_state_vector_fallback()
        actual = state.to_state_vector()
        np.testing.assert_allclose(actual, fallback, atol=1e-12)
        np.testing.assert_allclose(state.state_vector(), actual, atol=1e-12)


def test_to_state_vector_fallback_when_numba_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import cirq.sim.clifford.stabilizer_state_ch_form as ch_form_module

    # Test with n >= 8 (where Numba would normally be attempted)
    state = cirq.StabilizerStateChForm(8, initial_state=42)
    state.apply_h(0)
    state.apply_cx(0, 1)

    expected = state._to_state_vector_fallback()

    monkeypatch.setattr(ch_form_module, '_get_ch_to_state_vector_numba', lambda: None)
    actual_fallback = state.to_state_vector()
    np.testing.assert_allclose(actual_fallback, expected, atol=1e-12)


def test_to_state_vector_opt_out_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('numba')
    import cirq.sim.clifford.stabilizer_state_ch_form as ch_form_module

    state = cirq.StabilizerStateChForm(8, initial_state=17)
    state.apply_h(0)
    state.apply_cx(0, 1)

    expected = state._to_state_vector_fallback()

    called = False
    original_getter = ch_form_module._get_ch_to_state_vector_numba

    def spy_getter():
        nonlocal called
        called = True
        return original_getter()

    monkeypatch.setattr(ch_form_module, '_get_ch_to_state_vector_numba', spy_getter)
    monkeypatch.setenv('CIRQ_DISABLE_NUMBA', '1')

    actual = state.to_state_vector()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not called, 'Numba getter should not be called when CIRQ_DISABLE_NUMBA=1'


@pytest.mark.parametrize('n', [8, 9, 10])
def test_to_state_vector_numba_equivalence(n: int) -> None:
    pytest.importorskip('numba')
    import cirq.sim.clifford.stabilizer_state_ch_form as ch_form_module

    numba_fn = ch_form_module._get_ch_to_state_vector_numba()
    assert numba_fn is not None

    for init in [0, 1, 2**n - 1]:
        state = cirq.StabilizerStateChForm(n, initial_state=init)
        state.apply_h(0)
        state.apply_cx(0, 1)
        state.apply_z(1, exponent=0.5)
        if n > 2:
            state.apply_cz(1, 2)

        fallback = state._to_state_vector_fallback()
        actual = state.to_state_vector()
        direct_numba = numba_fn(
            state.n, state.F, state.M, state.gamma, state.v, state.s, complex(state.omega)
        )

        np.testing.assert_allclose(actual, fallback, atol=1e-12)
        np.testing.assert_allclose(actual, direct_numba, atol=1e-12)
        np.testing.assert_allclose(state.state_vector(), actual, atol=1e-12)


@pytest.mark.parametrize('n', [0, 1, 2, 4])
def test_to_state_vector_numba_direct_small_sizes(n: int) -> None:
    """Verifies that the Numba kernel handles small sizes (including n=0) correctly."""
    pytest.importorskip('numba')
    import cirq.sim.clifford.stabilizer_state_ch_form as ch_form_module

    numba_fn = ch_form_module._get_ch_to_state_vector_numba()
    assert numba_fn is not None

    state = cirq.StabilizerStateChForm(n)
    fallback = state._to_state_vector_fallback()
    direct_numba = numba_fn(
        state.n, state.F, state.M, state.gamma, state.v, state.s, complex(state.omega)
    )
    np.testing.assert_allclose(direct_numba, fallback, atol=1e-12)


def test_lazy_numba_import_isolated() -> None:
    code = (
        'import sys\n'
        'import cirq\n'
        'assert "numba" not in sys.modules\n'
        'state = cirq.StabilizerStateChForm(4)\n'
        'vec = state.to_state_vector()\n'
        'assert "numba" not in sys.modules\n'
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f'STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'


def test_to_state_vector_random_clifford() -> None:
    rng = np.random.RandomState(42)
    for _ in range(10):
        n = rng.randint(1, 6)
        state = cirq.StabilizerStateChForm(n, initial_state=rng.randint(0, 2**n))
        for _ in range(10):
            gate = rng.choice(['x', 'y', 'z', 'h', 'cx', 'cz'])
            q1 = rng.randint(0, n)
            if gate == 'x':
                state.apply_x(q1, exponent=rng.choice([0.5, 1.0, 1.5]))
            elif gate == 'y':
                state.apply_y(q1, exponent=rng.choice([0.5, 1.0, 1.5]))
            elif gate == 'z':
                state.apply_z(q1, exponent=rng.choice([0.5, 1.0, 1.5]))
            elif gate == 'h':
                state.apply_h(q1)
            elif gate in ('cx', 'cz') and n > 1:
                q2 = rng.randint(0, n)
                while q2 == q1:
                    q2 = rng.randint(0, n)
                if gate == 'cx':
                    state.apply_cx(q1, q2)
                else:
                    state.apply_cz(q1, q2)

        fallback = state._to_state_vector_fallback()
        actual = state.to_state_vector()
        np.testing.assert_allclose(actual, fallback, atol=1e-12)
        # Verify normalized
        assert np.isclose(np.sum(np.abs(actual) ** 2), 1.0, atol=1e-12)

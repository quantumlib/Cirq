# Copyright 2019 The Cirq Developers
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

"""Tests sampling/simulation methods that delegate to appropriate simulators."""
import collections

import numpy as np
import pytest
import sympy

import cirq
import cirq.testing


def test_sample():
    q = cirq.NamedQubit('q')

    with pytest.raises(ValueError, match="no measurements"):
        cirq.sample(cirq.Circuit(cirq.X(q)))
    # Unitary.
    results = cirq.sample(cirq.Circuit(cirq.X(q), cirq.measure(q)))
    assert results.histogram(key=q) == collections.Counter({1: 1})

    # Intermediate measurements.
    results = cirq.sample(
        cirq.Circuit(
            cirq.measure(q, key='drop'),
            cirq.X(q),
            cirq.measure(q),
        )
    )
    assert results.histogram(key='drop') == collections.Counter({0: 1})
    assert results.histogram(key=q) == collections.Counter({1: 1})

    # Overdamped everywhere.
    results = cirq.sample(
        cirq.Circuit(
            cirq.measure(q, key='drop'),
            cirq.X(q),
            cirq.measure(q),
        ),
        noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1)),
    )
    assert results.histogram(key='drop') == collections.Counter({0: 1})
    assert results.histogram(key=q) == collections.Counter({0: 1})


def test_sample_seed_unitary():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q) ** 0.2, cirq.measure(q))
    result = cirq.sample(circuit, repetitions=10, seed=1234)
    measurements = result.measurements['q']
    assert np.all(
        measurements
        == [[False], [False], [False], [False], [False], [False], [False], [False], [True], [False]]
    )


def test_sample_seed_non_unitary():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.depolarize(0.5).on(q), cirq.measure(q))
    result = cirq.sample(circuit, repetitions=10, seed=1234)
    assert np.all(
        result.measurements['q']
        == [[False], [False], [False], [True], [True], [False], [False], [True], [True], [True]]
    )


def test_sample_sweep():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit(cirq.X(q), cirq.Y(q) ** sympy.Symbol('t'), cirq.measure(q))

    # Unitary.
    results = cirq.sample_sweep(c, cirq.Linspace('t', 0, 1, 2), repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({1: 3})
    assert results[1].histogram(key=q) == collections.Counter({0: 3})

    # Overdamped.
    c = cirq.Circuit(
        cirq.X(q), cirq.amplitude_damp(1).on(q), cirq.Y(q) ** sympy.Symbol('t'), cirq.measure(q)
    )
    results = cirq.sample_sweep(c, cirq.Linspace('t', 0, 1, 2), repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({0: 3})
    assert results[1].histogram(key=q) == collections.Counter({1: 3})

    # Overdamped everywhere.
    c = cirq.Circuit(cirq.X(q), cirq.Y(q) ** sympy.Symbol('t'), cirq.measure(q))
    results = cirq.sample_sweep(
        c,
        cirq.Linspace('t', 0, 1, 2),
        noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1)),
        repetitions=3,
    )
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({0: 3})
    assert results[1].histogram(key=q) == collections.Counter({0: 3})


def test_sample_sweep_seed():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q))

    results = cirq.sample_sweep(
        circuit, [cirq.ParamResolver({'t': 0.5})] * 3, repetitions=2, seed=1234
    )
    assert np.all(results[0].measurements['q'] == [[False], [True]])
    assert np.all(results[1].measurements['q'] == [[False], [True]])
    assert np.all(results[2].measurements['q'] == [[True], [False]])

    results = cirq.sample_sweep(
        circuit,
        [cirq.ParamResolver({'t': 0.5})] * 3,
        repetitions=2,
        seed=np.random.RandomState(1234),
    )
    assert np.all(results[0].measurements['q'] == [[False], [True]])
    assert np.all(results[1].measurements['q'] == [[False], [True]])
    assert np.all(results[2].measurements['q'] == [[True], [False]])


def test_final_state_vector_different_program_types():
    a, b = cirq.LineQubit.range(2)

    np.testing.assert_allclose(cirq.final_state_vector(cirq.X), [0, 1], atol=1e-8)

    ops = [cirq.H(a), cirq.CNOT(a, b)]

    np.testing.assert_allclose(
        cirq.final_state_vector(ops), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-8
    )

    np.testing.assert_allclose(
        cirq.final_state_vector(cirq.Circuit(ops)), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-8
    )


def test_final_state_vector_initial_state():
    np.testing.assert_allclose(cirq.final_state_vector(cirq.X, initial_state=0), [0, 1], atol=1e-8)

    np.testing.assert_allclose(cirq.final_state_vector(cirq.X, initial_state=1), [1, 0], atol=1e-8)

    np.testing.assert_allclose(
        cirq.final_state_vector(cirq.X, initial_state=[np.sqrt(0.5), 1j * np.sqrt(0.5)]),
        [1j * np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-8,
    )


def test_final_state_vector_dtype_insensitive_to_initial_state():
    assert (
        cirq.final_state_vector(
            cirq.X,
        ).dtype
        == np.complex64
    )

    assert cirq.final_state_vector(cirq.X, initial_state=0).dtype == np.complex64

    assert (
        cirq.final_state_vector(cirq.X, initial_state=[np.sqrt(0.5), np.sqrt(0.5)]).dtype
        == np.complex64
    )

    assert (
        cirq.final_state_vector(cirq.X, initial_state=np.array([np.sqrt(0.5), np.sqrt(0.5)])).dtype
        == np.complex64
    )

    for t in [np.int32, np.float32, np.float64, np.complex64]:
        assert (
            cirq.final_state_vector(cirq.X, initial_state=np.array([1, 0], dtype=t)).dtype
            == np.complex64
        )

        assert (
            cirq.final_state_vector(
                cirq.X, initial_state=np.array([1, 0], dtype=t), dtype=np.complex128
            ).dtype
            == np.complex128
        )


def test_final_state_vector_param_resolver():
    s = sympy.Symbol('s')

    with pytest.raises(ValueError, match='not unitary'):
        _ = cirq.final_state_vector(cirq.X**s)

    np.testing.assert_allclose(
        cirq.final_state_vector(cirq.X**s, param_resolver={s: 0.5}), [0.5 + 0.5j, 0.5 - 0.5j]
    )


def test_final_state_vector_qubit_order():
    a, b = cirq.LineQubit.range(2)

    np.testing.assert_allclose(
        cirq.final_state_vector([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[a, b]),
        [0, 0, 0.5 + 0.5j, 0.5 - 0.5j],
    )

    np.testing.assert_allclose(
        cirq.final_state_vector([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[b, a]),
        [0, 0.5 + 0.5j, 0, 0.5 - 0.5j],
    )


def test_final_state_vector_seed():
    a = cirq.LineQubit(0)
    np.testing.assert_allclose(
        cirq.final_state_vector([cirq.X(a) ** 0.5, cirq.measure(a)], seed=123),
        [0, 0.707107 - 0.707107j],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        cirq.final_state_vector([cirq.X(a) ** 0.5, cirq.measure(a)], seed=124),
        [0.707107 + 0.707107j, 0],
        atol=1e-4,
    )


@pytest.mark.parametrize('repetitions', (0, 1, 100))
def test_repetitions(repetitions):
    a = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.H(a), cirq.measure(a, key='m'))
    r = cirq.sample(c, repetitions=repetitions)
    samples = r.data['m'].to_numpy()
    assert samples.shape == (repetitions,)
    assert np.issubdtype(samples.dtype, np.integer)


def test_final_density_matrix_different_program_types():
    a, b = cirq.LineQubit.range(2)

    np.testing.assert_allclose(cirq.final_density_matrix(cirq.X), [[0, 0], [0, 1]], atol=1e-8)

    ops = [cirq.H(a), cirq.CNOT(a, b)]

    np.testing.assert_allclose(
        cirq.final_density_matrix(cirq.Circuit(ops)),
        [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
        atol=1e-8,
    )


def test_final_density_matrix_initial_state():
    np.testing.assert_allclose(
        cirq.final_density_matrix(cirq.X, initial_state=0), [[0, 0], [0, 1]], atol=1e-8
    )

    np.testing.assert_allclose(
        cirq.final_density_matrix(cirq.X, initial_state=1), [[1, 0], [0, 0]], atol=1e-8
    )

    np.testing.assert_allclose(
        cirq.final_density_matrix(cirq.X, initial_state=[np.sqrt(0.5), 1j * np.sqrt(0.5)]),
        [[0.5, 0.5j], [-0.5j, 0.5]],
        atol=1e-8,
    )


def test_final_density_matrix_dtype_insensitive_to_initial_state():
    assert (
        cirq.final_density_matrix(
            cirq.X,
        ).dtype
        == np.complex64
    )

    assert cirq.final_density_matrix(cirq.X, initial_state=0).dtype == np.complex64

    assert (
        cirq.final_density_matrix(cirq.X, initial_state=[np.sqrt(0.5), np.sqrt(0.5)]).dtype
        == np.complex64
    )

    assert (
        cirq.final_density_matrix(
            cirq.X, initial_state=np.array([np.sqrt(0.5), np.sqrt(0.5)])
        ).dtype
        == np.complex64
    )

    for t in [np.int32, np.float32, np.float64, np.complex64]:
        assert (
            cirq.final_density_matrix(cirq.X, initial_state=np.array([1, 0], dtype=t)).dtype
            == np.complex64
        )

        assert (
            cirq.final_density_matrix(
                cirq.X, initial_state=np.array([1, 0], dtype=t), dtype=np.complex128
            ).dtype
            == np.complex128
        )


def test_final_density_matrix_param_resolver():
    s = sympy.Symbol('s')

    with pytest.raises(ValueError, match='not specified in parameter sweep'):
        _ = cirq.final_density_matrix(cirq.X**s)

    np.testing.assert_allclose(
        cirq.final_density_matrix(cirq.X**s, param_resolver={s: 0.5}),
        [[0.5 - 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 - 0.0j]],
    )


def test_final_density_matrix_qubit_order():
    a, b = cirq.LineQubit.range(2)

    np.testing.assert_allclose(
        cirq.final_density_matrix([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[a, b]),
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.5j], [0, 0, -0.5j, 0.5]],
    )

    np.testing.assert_allclose(
        cirq.final_density_matrix([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[b, a]),
        [[0, 0, 0, 0], [0, 0.5, 0, 0.5j], [0, 0, 0, 0], [0, -0.5j, 0, 0.5]],
    )

    np.testing.assert_allclose(
        cirq.final_density_matrix(
            [cirq.X(a), cirq.X(b) ** 0.5],
            qubit_order=[b, a],
            noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1.0)),
        ),
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    )


def test_final_density_matrix_seed_with_dephasing():
    a = cirq.LineQubit(0)
    np.testing.assert_allclose(
        cirq.final_density_matrix([cirq.X(a) ** 0.5, cirq.measure(a)], seed=123),
        [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        cirq.final_density_matrix([cirq.X(a) ** 0.5, cirq.measure(a)], seed=124),
        [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]],
        atol=1e-4,
    )


def test_final_density_matrix_seed_with_collapsing():
    a = cirq.LineQubit(0)
    np.testing.assert_allclose(
        cirq.final_density_matrix(
            [cirq.X(a) ** 0.5, cirq.measure(a)], seed=123, ignore_measurement_results=False
        ),
        [[0, 0], [0, 1]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        cirq.final_density_matrix(
            [cirq.X(a) ** 0.5, cirq.measure(a)], seed=124, ignore_measurement_results=False
        ),
        [[1, 0], [0, 0]],
        atol=1e-4,
    )


def test_final_density_matrix_noise():
    a = cirq.LineQubit(0)
    np.testing.assert_allclose(
        cirq.final_density_matrix([cirq.H(a), cirq.Z(a), cirq.H(a), cirq.measure(a)]),
        [[0, 0], [0, 1]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        cirq.final_density_matrix(
            [cirq.H(a), cirq.Z(a), cirq.H(a), cirq.measure(a)],
            noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1.0)),
        ),
        [[1, 0], [0, 0]],
        atol=1e-4,
    )


def test_ps_initial_state_wfn():
    q0, q1 = cirq.LineQubit.range(2)
    s00 = cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)
    sp0 = cirq.KET_PLUS(q0) * cirq.KET_ZERO(q1)

    np.testing.assert_allclose(
        cirq.final_state_vector(cirq.Circuit(cirq.I.on_each(q0, q1))),
        cirq.final_state_vector(cirq.Circuit(cirq.I.on_each(q0, q1)), initial_state=s00),
    )

    np.testing.assert_allclose(
        cirq.final_state_vector(cirq.Circuit(cirq.H(q0), cirq.I(q1))),
        cirq.final_state_vector(cirq.Circuit(cirq.I.on_each(q0, q1)), initial_state=sp0),
    )


def test_ps_initial_state_dmat():
    q0, q1 = cirq.LineQubit.range(2)
    s00 = cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)
    sp0 = cirq.KET_PLUS(q0) * cirq.KET_ZERO(q1)

    np.testing.assert_allclose(
        cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1))),
        cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1)), initial_state=s00),
    )

    np.testing.assert_allclose(
        cirq.final_density_matrix(cirq.Circuit(cirq.H(q0), cirq.I(q1))),
        cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1)), initial_state=sp0),
    )

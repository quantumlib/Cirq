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


def test_sample():
    q = cirq.NamedQubit('q')

    with pytest.raises(ValueError, match="no measurements"):
        cirq.sample(cirq.Circuit.from_ops(cirq.X(q)))
    # Unitary.
    results = cirq.sample(cirq.Circuit.from_ops(cirq.X(q), cirq.measure(q)))
    assert results.histogram(key=q) == collections.Counter({1: 1})

    # Intermediate measurements.
    results = cirq.sample(cirq.Circuit.from_ops(
        cirq.measure(q, key='drop'),
        cirq.X(q),
        cirq.measure(q),
    ))
    assert results.histogram(key='drop') == collections.Counter({0: 1})
    assert results.histogram(key=q) == collections.Counter({1: 1})

    # Overdamped everywhere.
    results = cirq.sample(cirq.Circuit.from_ops(
        cirq.measure(q, key='drop'),
        cirq.X(q),
        cirq.measure(q),
    ),
                          noise=cirq.ConstantQubitNoiseModel(
                              cirq.amplitude_damp(1)))
    assert results.histogram(key='drop') == collections.Counter({0: 1})
    assert results.histogram(key=q) == collections.Counter({0: 1})


def test_sample_sweep():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.Y(q)**sympy.Symbol('t'),
        cirq.measure(q))

    # Unitary.
    results = cirq.sample_sweep(c, cirq.Linspace('t', 0, 1, 2), repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({1: 3})
    assert results[1].histogram(key=q) == collections.Counter({0: 3})

    # Overdamped.
    c = cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.amplitude_damp(1).on(q),
        cirq.Y(q)**sympy.Symbol('t'),
        cirq.measure(q))
    results = cirq.sample_sweep(
        c,
        cirq.Linspace('t', 0, 1, 2),
        repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({0: 3})
    assert results[1].histogram(key=q) == collections.Counter({1: 3})

    # Overdamped everywhere.
    c = cirq.Circuit.from_ops(cirq.X(q),
                              cirq.Y(q)**sympy.Symbol('t'), cirq.measure(q))
    results = cirq.sample_sweep(c,
                                cirq.Linspace('t', 0, 1, 2),
                                noise=cirq.ConstantQubitNoiseModel(
                                    cirq.amplitude_damp(1)),
                                repetitions=3)
    assert len(results) == 2
    assert results[0].histogram(key=q) == collections.Counter({0: 3})
    assert results[1].histogram(key=q) == collections.Counter({0: 3})


def test_final_wavefunction_different_program_types():
    a, b = cirq.LineQubit.range(2)

    np.testing.assert_allclose(cirq.final_wavefunction(cirq.X), [0, 1],
                               atol=1e-8)

    ops = [cirq.H(a), cirq.CNOT(a, b)]

    np.testing.assert_allclose(
        cirq.final_wavefunction(ops),
        [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.final_wavefunction(cirq.Circuit.from_ops(ops)),
        [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.final_wavefunction(
            cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                           cirq.Circuit.from_ops(ops))),
        [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        atol=1e-8)


def test_final_wavefunction_initial_state():
    np.testing.assert_allclose(cirq.final_wavefunction(cirq.X, initial_state=0),
                               [0, 1],
                               atol=1e-8)

    np.testing.assert_allclose(cirq.final_wavefunction(cirq.X, initial_state=1),
                               [1, 0],
                               atol=1e-8)

    np.testing.assert_allclose(
        cirq.final_wavefunction(cirq.X,
                                initial_state=[np.sqrt(0.5),
                                               1j * np.sqrt(0.5)]),
        [1j * np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-8)


def test_final_wavefunction_dtype_insensitive_to_initial_state():
    assert cirq.final_wavefunction(cirq.X,).dtype == np.complex64

    assert cirq.final_wavefunction(cirq.X,
                                   initial_state=0).dtype == np.complex64

    assert cirq.final_wavefunction(cirq.X,
                                   initial_state=[np.sqrt(0.5),
                                                  np.sqrt(0.5)
                                                 ]).dtype == np.complex64

    assert cirq.final_wavefunction(cirq.X,
                                   initial_state=np.array(
                                       [np.sqrt(0.5),
                                        np.sqrt(0.5)])).dtype == np.complex64

    for t in [np.int32, np.float32, np.float64, np.complex64]:
        assert cirq.final_wavefunction(
            cirq.X, initial_state=np.array([1, 0],
                                           dtype=t)).dtype == np.complex64

        assert cirq.final_wavefunction(
            cirq.X,
            initial_state=np.array([1, 0], dtype=t),
            dtype=np.complex128).dtype == np.complex128


def test_final_wavefunction_param_resolver():
    s = sympy.Symbol('s')

    with pytest.raises(ValueError, match='not unitary'):
        _ = cirq.final_wavefunction(cirq.X**s)

    np.testing.assert_allclose(
        cirq.final_wavefunction(cirq.X**s, param_resolver={s: 0.5}),
        [0.5 + 0.5j, 0.5 - 0.5j])


def test_final_wavefunction_qubit_order():
    a, b = cirq.LineQubit.range(2)

    np.testing.assert_allclose(
        cirq.final_wavefunction([cirq.X(a), cirq.X(b)**0.5], qubit_order=[a,
                                                                          b]),
        [0, 0, 0.5 + 0.5j, 0.5 - 0.5j])

    np.testing.assert_allclose(
        cirq.final_wavefunction([cirq.X(a), cirq.X(b)**0.5], qubit_order=[b,
                                                                          a]),
        [0, 0.5 + 0.5j, 0, 0.5 - 0.5j])

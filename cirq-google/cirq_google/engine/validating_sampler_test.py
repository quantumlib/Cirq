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
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import sympy

import cirq
import cirq_google as cg


def test_device_validation():
    sampler = cg.ValidatingSampler(
        device=cg.Sycamore23, validator=lambda c, s, r: True, sampler=cirq.Simulator()
    )

    # Good qubit
    q = cirq.GridQubit(5, 2)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    results = sampler.run_sweep(circuit, sweep, repetitions=100)
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)

    # Bad qubit
    q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    with pytest.raises(ValueError, match='Qubit not on device'):
        results = sampler.run_sweep(circuit, sweep, repetitions=100)


def _batch_size_less_than_two(
    circuits: list[cirq.Circuit], sweeps: list[cirq.Sweepable], repetitions: int
):
    if len(circuits) > 2:
        raise ValueError('Too many batches')


def test_batch_validation():
    sampler = cg.ValidatingSampler(
        device=cirq.UNCONSTRAINED_DEVICE,
        validator=_batch_size_less_than_two,
        sampler=cirq.Simulator(),
    )
    q = cirq.GridQubit(2, 2)
    circuits = [
        cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m')),
        cirq.Circuit(cirq.X(q) ** sympy.Symbol('x'), cirq.measure(q, key='m2')),
    ]
    sweeps = [cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1])]
    results = sampler.run_batch(circuits, sweeps, repetitions=100)

    assert np.all(results[0][0].measurements['m'] == 1)
    assert np.all(results[0][1].measurements['m'] == 0)
    assert np.all(results[1][0].measurements['m2'] == 0)
    assert np.all(results[1][1].measurements['m2'] == 1)

    circuits = [
        cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m')),
        cirq.Circuit(cirq.X(q) ** sympy.Symbol('x'), cirq.measure(q, key='m2')),
        cirq.Circuit(cirq.measure(q, key='m3')),
    ]
    sweeps = [cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1]), {}]
    with pytest.raises(ValueError, match='Too many batches'):
        results = sampler.run_batch(circuits, sweeps, repetitions=100)


def _too_many_reps(circuits: list[cirq.Circuit], sweeps: list[cirq.Sweepable], repetitions: int):
    if repetitions > 10000:
        raise ValueError('Too many repetitions')


def test_sweeps_validation():
    sampler = cg.ValidatingSampler(
        device=cirq.UNCONSTRAINED_DEVICE, validator=_too_many_reps, sampler=cirq.Simulator()
    )
    q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweeps = [cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1])]
    with pytest.raises(ValueError, match='Too many repetitions'):
        _ = sampler.run_sweep(circuit, sweeps, repetitions=20000)


def test_batch_default_sweeps():
    sampler = cg.ValidatingSampler()
    q = cirq.GridQubit(2, 2)
    circuits = [
        cirq.Circuit(cirq.X(q), cirq.measure(q, key='m')),
        cirq.Circuit(cirq.measure(q, key='m2')),
    ]
    results = sampler.run_batch(circuits, None, repetitions=100)
    assert np.all(results[0][0].measurements['m'] == 1)
    assert np.all(results[1][0].measurements['m2'] == 0)


class _BackCompatSampler(cirq.Sampler):
    """Sampler before run_sweep got the `prng` param."""

    def run_sweep(  # type: ignore[override]
        self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence[cirq.Result]:
        return cirq.Simulator().run_sweep(program, params, repetitions)


def test_validating_sampler_without_prng_no_forwarding() -> None:
    """`run_sweep` and `run_batch` shouldn't forward `prng` when the caller didn't pass one in."""

    sampler = cg.ValidatingSampler(sampler=_BackCompatSampler())
    q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q, key='m'))
    expected = np.ones((2, 1, 1))

    np.testing.assert_equal(sampler.run_sweep(circuit, None, 2)[0].records['m'], expected)
    np.testing.assert_equal(sampler.run_batch([circuit], None, 2)[0][0].records['m'], expected)


def test_passing_prng_to_validating_sampler_without_prng_fails() -> None:
    """Samplers that don't use `prng` should throw an error upon receiving it."""

    sampler = cg.ValidatingSampler(sampler=_BackCompatSampler())
    q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q, key='m'))

    with pytest.raises(TypeError, match='run_sweep'):
        sampler.run_sweep(circuit, None, 2, prng=np.random.default_rng(0))
    with pytest.raises(TypeError, match='run_sweep'):
        sampler.run_batch([circuit], None, 2, prng=np.random.default_rng(0))

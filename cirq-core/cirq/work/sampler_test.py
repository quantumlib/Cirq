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
"""Tests for cirq.Sampler."""
import pytest

import numpy as np
import pandas as pd
import sympy

import cirq


@pytest.mark.asyncio
async def test_sampler_async_fail():
    class FailingSampler(cirq.Sampler):
        def run_sweep(self, program, params, repetitions: int = 1):
            raise ValueError('test')

    with pytest.raises(ValueError, match='test'):
        await FailingSampler().run_async(cirq.Circuit(), repetitions=1)

    with pytest.raises(ValueError, match='test'):
        await FailingSampler().run_sweep_async(cirq.Circuit(), repetitions=1, params=None)


def test_sampler_sample_multiple_params():
    a, b = cirq.LineQubit.range(2)
    s = sympy.Symbol('s')
    t = sympy.Symbol('t')
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a) ** s, cirq.X(b) ** t, cirq.measure(a, b, key='out'))
    results = sampler.sample(
        circuit,
        repetitions=3,
        params=[
            {'s': 0, 't': 0},
            {'s': 0, 't': 1},
            {'s': 1, 't': 0},
            {'s': 1, 't': 1},
        ],
    )
    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(
            columns=['s', 't', 'out'],
            index=[0, 1, 2] * 4,
            data=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
                [1, 0, 2],
                [1, 0, 2],
                [1, 0, 2],
                [1, 1, 3],
                [1, 1, 3],
                [1, 1, 3],
            ],
        ),
    )


def test_sampler_sample_sweep():
    a = cirq.LineQubit(0)
    t = sympy.Symbol('t')
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a) ** t, cirq.measure(a, key='out'))
    results = sampler.sample(circuit, repetitions=3, params=cirq.Linspace('t', 0, 2, 3))
    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(
            columns=['t', 'out'],
            index=[0, 1, 2] * 3,
            data=[
                [0.0, 0],
                [0.0, 0],
                [0.0, 0],
                [1.0, 1],
                [1.0, 1],
                [1.0, 1],
                [2.0, 0],
                [2.0, 0],
                [2.0, 0],
            ],
        ),
    )


def test_sampler_sample_no_params():
    a, b = cirq.LineQubit.range(2)
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a), cirq.measure(a, b, key='out'))
    results = sampler.sample(circuit, repetitions=3)
    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(
            columns=['out'],
            index=[0, 1, 2],
            data=[
                [2],
                [2],
                [2],
            ],
        ),
    )


def test_sampler_sample_inconsistent_keys():
    q = cirq.LineQubit(0)
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.measure(q, key='out'))
    with pytest.raises(ValueError, match='Inconsistent sweep parameters'):
        _ = sampler.sample(
            circuit,
            params=[
                {'a': 1},
                {'a': 1, 'b': 2},
            ],
        )


@pytest.mark.asyncio
async def test_sampler_async_not_run_inline():
    ran = False

    class S(cirq.Sampler):
        def run_sweep(self, *args, **kwargs):
            nonlocal ran
            ran = True
            return []

    a = S().run_sweep_async(cirq.Circuit(), params=None)
    assert not ran
    assert await a == []
    assert ran


def test_sampler_run_batch():
    sampler = cirq.ZerosSampler()
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a) ** sympy.Symbol('t'), cirq.measure(a, key='m'))
    circuit2 = cirq.Circuit(cirq.Y(a) ** sympy.Symbol('t'), cirq.measure(a, key='m'))
    params1 = cirq.Points('t', [0.3, 0.7])
    params2 = cirq.Points('t', [0.4, 0.6])
    results = sampler.run_batch(
        [circuit1, circuit2], params_list=[params1, params2], repetitions=[1, 2]
    )
    assert len(results) == 2
    for result, param in zip(results[0], [0.3, 0.7]):
        assert result.repetitions == 1
        assert result.params.param_dict == {'t': param}
        assert result.measurements == {'m': np.array([[0]], dtype='uint8')}
    for result, param in zip(results[1], [0.4, 0.6]):
        assert result.repetitions == 2
        assert result.params.param_dict == {'t': param}
        assert len(result.measurements) == 1
        assert np.array_equal(result.measurements['m'], np.array([[0], [0]], dtype='uint8'))


def test_sampler_run_batch_default_params_and_repetitions():
    sampler = cirq.ZerosSampler()
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a), cirq.measure(a, key='m'))
    circuit2 = cirq.Circuit(cirq.Y(a), cirq.measure(a, key='m'))
    results = sampler.run_batch([circuit1, circuit2])
    assert len(results) == 2
    for result_list in results:
        assert len(result_list) == 1
        result = result_list[0]
        assert result.repetitions == 1
        assert result.params.param_dict == {}
        assert result.measurements == {'m': np.array([[0]], dtype='uint8')}


def test_sampler_run_batch_bad_input_lengths():
    sampler = cirq.ZerosSampler()
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a) ** sympy.Symbol('t'), cirq.measure(a, key='m'))
    circuit2 = cirq.Circuit(cirq.Y(a) ** sympy.Symbol('t'), cirq.measure(a, key='m'))
    params1 = cirq.Points('t', [0.3, 0.7])
    params2 = cirq.Points('t', [0.4, 0.6])
    with pytest.raises(ValueError, match='2 and 1'):
        _ = sampler.run_batch([circuit1, circuit2], params_list=[params1])
    with pytest.raises(ValueError, match='2 and 3'):
        _ = sampler.run_batch(
            [circuit1, circuit2], params_list=[params1, params2], repetitions=[1, 2, 3]
        )

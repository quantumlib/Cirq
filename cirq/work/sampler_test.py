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
        await FailingSampler().run_sweep_async(cirq.Circuit(),
                                               repetitions=1,
                                               params=None)


def test_sampler_sample_multiple_params():
    a, b = cirq.LineQubit.range(2)
    s = sympy.Symbol('s')
    t = sympy.Symbol('t')
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(
        cirq.X(a)**s,
        cirq.X(b)**t, cirq.measure(a, b, key='out'))
    results = sampler.sample(circuit,
                             repetitions=3,
                             params=[
                                 {
                                     's': 0,
                                     't': 0
                                 },
                                 {
                                     's': 0,
                                     't': 1
                                 },
                                 {
                                     's': 1,
                                     't': 0
                                 },
                                 {
                                     's': 1,
                                     't': 1
                                 },
                             ])
    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(columns=['s', 't', 'out'],
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
                     ]))


def test_sampler_sample_sweep():
    a = cirq.LineQubit(0)
    t = sympy.Symbol('t')
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a)**t, cirq.measure(a, key='out'))
    results = sampler.sample(circuit,
                             repetitions=3,
                             params=cirq.Linspace('t', 0, 2, 3))
    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(columns=['t', 'out'],
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
                     ]))


def test_sampler_sample_no_params():
    a, b = cirq.LineQubit.range(2)
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.X(a), cirq.measure(a, b, key='out'))
    results = sampler.sample(circuit, repetitions=3)
    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(columns=['out'], index=[0, 1, 2], data=[
            [2],
            [2],
            [2],
        ]))


def test_sampler_sample_inconsistent_keys():
    q = cirq.LineQubit(0)
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.measure(q, key='out'))
    with pytest.raises(ValueError, match='Inconsistent sweep parameters'):
        _ = sampler.sample(circuit, params=[
            {
                'a': 1
            },
            {
                'a': 1,
                'b': 2
            },
        ])


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


def test_sampler_alternate_method_choices():

    def assert_sampler_methods_work(sampler: cirq.Sampler):
        a, b = cirq.LineQubit.range(2)
        c2 = cirq.Circuit(
            cirq.Moment(
                [cirq.measure(a, key='c'),
                 cirq.measure(b, key='out_t')]),
            cirq.Moment([cirq.measure(a, b, key='n')]),
        )
        dict_no_t = {'c': 1, 'n': 2, 'out_t': 0}
        dict_yes_t = {'c': 1, 'n': 2, 'out_t': 1}
        no_t_trial_result = cirq.TrialResult(params=cirq.ParamResolver(None),
                                             measurements={
                                                 'c':
                                                 np.array([[1], [1], [1]],
                                                          dtype=np.int64),
                                                 'out_t':
                                                 np.array([[0], [0], [0]],
                                                          dtype=np.int64),
                                                 'n':
                                                 np.array(
                                                     [[1, 0], [1, 0], [1, 0]],
                                                     dtype=np.int64),
                                             })
        yes_t_trial_result = cirq.TrialResult(
            params=cirq.ParamResolver({'t': 1}),
            measurements={
                'c': np.array([[1], [1], [1]], dtype=np.int64),
                'out_t': np.array([[1], [1], [1]], dtype=np.int64),
                'n': np.array([[1, 0], [1, 0], [1, 0]], dtype=np.int64),
            })

        assert sampler.sample_dict(c2) == dict_no_t
        assert sampler.sample_dict(c2, params={'t': 1}) == dict_yes_t

        assert sampler.sample_dicts(c2, repetitions=0, params={'t': 1}) == []
        assert sampler.sample_dicts(c2, repetitions=2) == [dict_no_t] * 2
        assert sampler.sample_dicts(c2, repetitions=3) == [dict_no_t] * 3
        assert sampler.sample_dicts(c2, repetitions=3,
                                    params={'t': 1}) == [dict_yes_t] * 3

        assert sampler.run(c2, repetitions=3,
                           param_resolver={'t': 1}) == yes_t_trial_result
        assert sampler.run_sweep(c2, repetitions=3, params=[{}, {}, {
            't': 1
        }]) == [
            no_t_trial_result,
            no_t_trial_result,
            yes_t_trial_result,
        ]

        pd.testing.assert_frame_equal(
            sampler.sample(c2, params=[{
                't': 0
            }, {
                't': 1
            }], repetitions=2),
            pd.DataFrame(index=pd.Int64Index([0, 1, 0, 1]),
                         columns=['t', 'c', 'out_t', 'n'],
                         data=[
                             [0, 1, 0, 2],
                             [0, 1, 0, 2],
                             [1, 1, 1, 2],
                             [1, 1, 1, 2],
                         ]))
        pd.testing.assert_frame_equal(
            sampler.sample(c2, params={}, repetitions=3),
            pd.DataFrame(index=pd.Int64Index([0, 1, 2]),
                         columns=['c', 'out_t', 'n'],
                         data=[
                             [1, 0, 2],
                             [1, 0, 2],
                             [1, 0, 2],
                         ]))

        # Check expected data dependency: n should be the circuit length.
        c3 = cirq.Circuit(
            cirq.Moment(
                [cirq.measure(a, key='c'),
                 cirq.measure(b, key='out_t')]),
            cirq.Moment([cirq.measure(a, b, key='n')]),
            cirq.Moment(),
        )
        pd.testing.assert_frame_equal(
            sampler.sample(c3, params={}, repetitions=3),
            pd.DataFrame(index=pd.Int64Index([0, 1, 2]),
                         columns=['c', 'out_t', 'n'],
                         data=[
                             [1, 0, 3],
                             [1, 0, 3],
                             [1, 0, 3],
                         ]))

    class AbstractSampler(cirq.Sampler):
        pass

    with pytest.raises(TypeError, match="abstract class"):
        _ = AbstractSampler()

    class SamplerViaSampleDict(cirq.Sampler):

        def sample_dict(self, program, *, params=None):
            t = cirq.ParamResolver(params).param_dict.get('t', 0)
            return {'c': 1, 'n': len(program), 'out_t': t}

    assert_sampler_methods_work(SamplerViaSampleDict())

    class SamplerViaSampleDicts(cirq.Sampler):

        def sample_dicts(self, program, *, repetitions=1, params=None):
            t = cirq.ParamResolver(params).param_dict.get('t', 0)
            return [{'c': 1, 'n': len(program), 'out_t': t}] * repetitions

    assert_sampler_methods_work(SamplerViaSampleDicts())

    class SamplerViaRun(cirq.Sampler):

        def run(self, program, param_resolver=None, repetitions=1):
            param_resolver = cirq.ParamResolver(param_resolver)
            t = param_resolver.param_dict.get('t', 0)
            b = cirq.big_endian_int_to_bits(len(program), bit_count=2)
            return cirq.TrialResult(params=param_resolver,
                                    measurements={
                                        'c': np.array([[1]] * repetitions),
                                        'out_t': np.array([[t]] * repetitions),
                                        'n': np.array([b] * repetitions),
                                    })

    assert_sampler_methods_work(SamplerViaRun())

    class SamplerViaRunSweep(cirq.Sampler):

        def run_sweep(self, program, params=None, repetitions=1):
            return [
                SamplerViaRun().run(program, param, repetitions)
                for param in cirq.to_sweep(params)
            ]

    assert_sampler_methods_work(SamplerViaRunSweep())

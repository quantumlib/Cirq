# Copyright 2018 The Cirq Developers
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
"""Tests for sweepable.py."""

import itertools

import pytest

import cirq


def test_to_resolvers_single():
    resolver = cirq.ParamResolver({})
    assert cirq.to_resolvers(resolver) == [resolver]


def test_to_resolvers_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert cirq.to_resolvers(sweep) == list(sweep)


def test_to_resolvers_iterable():
    resolvers = [cirq.ParamResolver({'a': 2}), cirq.ParamResolver({'a': 1})]
    assert cirq.to_resolvers(resolvers) == resolvers


def test_to_resolvers_iterable_sweeps():
    sweeps = [cirq.Linspace('a', 0, 1, 10), cirq.Linspace('b', 0, 1, 10)]
    assert cirq.to_resolvers(sweeps) == list(itertools.chain(*sweeps))


def test_to_sweeps_single():
    resolver = cirq.ParamResolver({})
    assert cirq.study.to_sweeps(resolver) == [cirq.UnitSweep]


def test_to_sweeps_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert cirq.study.to_sweeps(sweep) == [sweep]


def test_to_sweeps_iterable():
    resolvers = [cirq.ParamResolver({'a': 2}), cirq.ParamResolver({'a': 1})]
    assert cirq.study.to_sweeps(resolvers) == [
        cirq.study.Zip(cirq.Points('a', [2])),
        cirq.study.Zip(cirq.Points('a', [1])),
    ]


def test_to_sweeps_iterable_sweeps():
    sweeps = [cirq.Linspace('a', 0, 1, 10), cirq.Linspace('b', 0, 1, 10)]
    assert cirq.study.to_sweeps(sweeps) == sweeps


def test_to_sweeps_invalid():
    with pytest.raises(TypeError):
        cirq.study.to_sweeps('nope')

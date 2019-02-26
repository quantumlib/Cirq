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

"""Sampling/simulation methods that delegate to appropriate simulators."""
import collections
from typing import List, Type, Union, Optional

import numpy as np

import cirq


def test_run():
    q = cirq.NamedQubit('q')

    # Unitary.
    results = cirq.run(cirq.Circuit.from_ops(cirq.X(q), cirq.measure(q)))
    assert results.histogram(key=q) == collections.Counter({1: 1})

    # Intermediate measurements.
    results = cirq.run(cirq.Circuit.from_ops(
        cirq.measure(q, key='drop'),
        cirq.X(q),
        cirq.measure(q),
    ))
    assert results.histogram(key='drop') == collections.Counter({0: 1})
    assert results.histogram(key=q) == collections.Counter({1: 1})

    # Noisy.
    results = cirq.run(cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.measure(q),
    ), noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1)))
    assert results.histogram(key=q) == collections.Counter({0: 1})


def test_run_sweep():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.Z(q)**cirq.Symbol('t'),
        cirq.measure(q))

    # Unitary.
    results = cirq.run_sweep(c, cirq.Linspace('t', 0, 1, 5), repetitions=3)
    assert len(results) == 5
    for result in results:
        assert result.histogram(key=q) == collections.Counter({1: 3})

    # Overdamped.
    results = cirq.run_sweep(
        c,
        cirq.Linspace('t', 0, 1, 5),
        noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(1)),
        repetitions=3)
    assert len(results) == 5
    for result in results:
        assert result.histogram(key=q) == collections.Counter({0: 3})

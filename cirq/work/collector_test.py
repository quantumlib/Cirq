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

import abc
from typing import Optional, Any, TypeVar, NamedTuple, Iterable, Union, List

import numpy as np

import cirq


def test_circuit_sample_job_equality():
    eq = cirq.testing.EqualsTester()
    c1 = cirq.Circuit()
    c2 = cirq.Circuit.from_ops(cirq.measure(cirq.LineQubit(0)))

    eq.add_equality_group(
        cirq.CircuitSampleJob(c1, repetitions=10),
        cirq.CircuitSampleJob(c1, repetitions=10, id=None)
    )
    eq.add_equality_group(cirq.CircuitSampleJob(c2, repetitions=10))
    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=100))
    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=10, id='test'))


def test_circuit_sample_job_repr():
    cirq.testing.assert_equivalent_repr(cirq.CircuitSampleJob(
        cirq.Circuit.from_ops(cirq.H(cirq.LineQubit(0))),
        repetitions=10,
        id='guess'))


def test_async_collect():
    class ManualCollector(cirq.SampleCollector):
        def next_job(self):
            q = cirq.LineQubit(0)
            return cirq.CircuitSampleJob(
                circuit=cirq.Circuit.from_ops(
                    cirq.H(q),
                    cirq.measure(q)),
                repetitions=10,
                id='test')

        def on_job_result(self, job, result):
            assert job.id == 'test'
            assert False
            pass

    m = ManualCollector()

    r = cirq.async_collect_samples(
        m,
        sampler=cirq.Simulator())
    cirq.testing.assert_asyncio_will_have_result(r, None)
    print(r)
    assert False

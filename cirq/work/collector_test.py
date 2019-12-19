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
import pytest

import cirq


def test_circuit_sample_job_equality():
    eq = cirq.testing.EqualsTester()
    c1 = cirq.Circuit()
    c2 = cirq.Circuit(cirq.measure(cirq.LineQubit(0)))

    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=10),
                          cirq.CircuitSampleJob(c1, repetitions=10, tag=None))
    eq.add_equality_group(cirq.CircuitSampleJob(c2, repetitions=10))
    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=100))
    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=10, tag='test'))


def test_circuit_sample_job_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.CircuitSampleJob(cirq.Circuit(cirq.H(cirq.LineQubit(0))),
                              repetitions=10,
                              tag='guess'))


@pytest.mark.asyncio
async def test_async_collect():
    received = []

    class TestCollector(cirq.Collector):

        def next_job(self):
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
            return cirq.CircuitSampleJob(circuit=circuit,
                                         repetitions=10,
                                         tag='test')

        def on_job_result(self, job, result):
            received.append(job.tag)

    completion = TestCollector().collect_async(sampler=cirq.Simulator(),
                                               max_total_samples=100,
                                               concurrency=5)
    assert await completion is None
    assert received == ['test'] * 10


def test_collect():
    received = []

    class TestCollector(cirq.Collector):

        def next_job(self):
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
            return cirq.CircuitSampleJob(circuit=circuit,
                                         repetitions=10,
                                         tag='test')

        def on_job_result(self, job, result):
            received.append(job.tag)

    TestCollector().collect(sampler=cirq.Simulator(),
                            max_total_samples=100,
                            concurrency=5)
    assert received == ['test'] * 10


def test_collect_with_reaction():
    events = [0]
    sent = 0
    received = 0

    class TestCollector(cirq.Collector):

        def next_job(self):
            nonlocal sent
            if sent >= received + 3:
                return None
            sent += 1
            events.append(sent)
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
            return cirq.CircuitSampleJob(circuit=circuit,
                                         repetitions=10,
                                         tag=sent)

        def on_job_result(self, job, result):
            nonlocal received
            received += 1
            events.append(-job.tag)

    TestCollector().collect(sampler=cirq.Simulator(),
                            max_total_samples=100,
                            concurrency=5)
    # Expected sends and receives are present.
    assert sorted(events) == list(range(-10, 1 + 10))
    # Sends are in order.
    assert [e for e in events if e > 0] == list(range(1, 11))
    # Every receive comes after the corresponding send.
    assert all(events.index(-k) > events.index(k) for k in range(1, 11))


def test_flatten_jobs_terminate_from_collector():
    sent = False
    received = []

    class TestCollector(cirq.Collector):

        def next_job(self):
            nonlocal sent
            if sent:
                return
            sent = True
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
            a = cirq.CircuitSampleJob(circuit=circuit,
                                      repetitions=10,
                                      tag='test')
            b = cirq.CircuitSampleJob(circuit=circuit,
                                      repetitions=10,
                                      tag='test')
            return [[a, None], [[[None]]], [[[]]], b]

        def on_job_result(self, job, result):
            received.append(job.tag)

    TestCollector().collect(sampler=cirq.Simulator(), concurrency=5)
    assert received == ['test'] * 2

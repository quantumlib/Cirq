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
"""Tests for SimulatedLocalJob."""
import pytest
import numpy as np
import sympy

import cirq
from cirq_google.engine.client import quantum
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType

Q = cirq.GridQubit(2, 2)


class ParentProgram(AbstractLocalProgram):
    def delete(self, delete_jobs: bool = False) -> None:
        pass

    def delete_job(self, program_id: str) -> None:
        pass


def test_run():
    program = ParentProgram([cirq.Circuit(cirq.X(Q), cirq.measure(Q, key='m'))], None)
    job = SimulatedLocalJob(
        job_id='test_job', processor_id='test1', parent_program=program, repetitions=100, sweeps=[]
    )
    assert job.id() == 'test_job'
    assert job.execution_status() == quantum.enums.ExecutionStatus.State.READY
    results = job.results()
    assert np.all(results[0].measurements['m'] == 1)
    assert job.execution_status() == quantum.enums.ExecutionStatus.State.SUCCESS


def test_run_sweep():
    program = ParentProgram(
        [cirq.Circuit(cirq.X(Q) ** sympy.Symbol('t'), cirq.measure(Q, key='m'))], None
    )
    job = SimulatedLocalJob(
        job_id='test_job',
        processor_id='test1',
        parent_program=program,
        repetitions=100,
        sweeps=[cirq.Points(key='t', points=[1, 0])],
    )
    assert job.execution_status() == quantum.enums.ExecutionStatus.State.READY
    results = job.results()
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)
    assert job.execution_status() == quantum.enums.ExecutionStatus.State.SUCCESS


def test_run_batch():
    program = ParentProgram(
        [
            cirq.Circuit(cirq.X(Q) ** sympy.Symbol('t'), cirq.measure(Q, key='m')),
            cirq.Circuit(cirq.X(Q) ** sympy.Symbol('x'), cirq.measure(Q, key='m2')),
        ],
        None,
    )
    job = SimulatedLocalJob(
        job_id='test_job',
        processor_id='test1',
        parent_program=program,
        repetitions=100,
        sweeps=[cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1])],
    )
    assert job.execution_status() == quantum.enums.ExecutionStatus.State.READY
    results = job.batched_results()
    assert np.all(results[0][0].measurements['m'] == 1)
    assert np.all(results[0][1].measurements['m'] == 0)
    assert np.all(results[1][0].measurements['m2'] == 0)
    assert np.all(results[1][1].measurements['m2'] == 1)
    assert job.execution_status() == quantum.enums.ExecutionStatus.State.SUCCESS


def test_cancel():
    program = ParentProgram([cirq.Circuit(cirq.X(Q), cirq.measure(Q, key='m'))], None)
    job = SimulatedLocalJob(
        job_id='test_job', processor_id='test1', parent_program=program, repetitions=100, sweeps=[]
    )
    job.cancel()
    assert job.execution_status() == quantum.enums.ExecutionStatus.State.CANCELLED


def test_unsupported_types():
    program = ParentProgram([cirq.Circuit(cirq.X(Q), cirq.measure(Q, key='m'))], None)
    job = SimulatedLocalJob(
        job_id='test_job',
        processor_id='test1',
        parent_program=program,
        repetitions=100,
        sweeps=[],
        simulation_type=LocalSimulationType.ASYNCHRONOUS,
    )
    with pytest.raises(ValueError, match='Unsupported simulation type'):
        job.results()
    with pytest.raises(ValueError, match='Unsupported simulation type'):
        job.batched_results()


def test_failure():
    program = ParentProgram(
        [cirq.Circuit(cirq.X(Q) ** sympy.Symbol('t'), cirq.measure(Q, key='m'))], None
    )
    job = SimulatedLocalJob(
        job_id='test_job',
        processor_id='test1',
        parent_program=program,
        repetitions=100,
        sweeps=[cirq.Points(key='x', points=[1, 0])],
    )
    try:
        _ = job.results()
    except ValueError:
        code, message = job.failure()
        assert code == '500'
        assert 'Circuit contains ops whose symbols were not specified' in message

    try:
        _ = job.batched_results()
    except ValueError:
        code, message = job.failure()
        assert code == '500'
        assert 'Circuit contains ops whose symbols were not specified' in message


def test_run_calibration_unsupported():
    program = ParentProgram([cirq.Circuit()], None)
    job = SimulatedLocalJob(
        job_id='test_job',
        processor_id='test1',
        parent_program=program,
        repetitions=100,
        sweeps=[{}],
    )
    with pytest.raises(NotImplementedError):
        _ = job.calibration_results()

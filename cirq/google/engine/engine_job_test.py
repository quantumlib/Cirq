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

from unittest import mock

import pytest

from cirq.google.engine.client.quantum_v1alpha1 import types as qtypes

import cirq
import cirq.google as cg


def test_status():
    engine = mock.Mock()
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve',
                            execution_status=qtypes.ExecutionStatus(
                                state=qtypes.ExecutionStatus.State.RUNNING))
    engine.get_job.return_value = job
    job = cg.EngineJob(job_config=cg.JobConfig(job_id='steve'),
                       job=job,
                       engine=engine)
    assert job.status() == 'RUNNING'
    engine.get_job.assert_called_once()


def test_get_calibration():
    engine = mock.Mock()
    job = qtypes.QuantumJob(
        name='projects/a/programs/b/jobs/steve',
        execution_status=qtypes.ExecutionStatus(calibration_name='hobbes'))
    calibration = mock.Mock()
    engine.get_calibration.return_value = calibration
    job = cg.EngineJob(job_config=cg.JobConfig(job_id='steve'),
                       job=job,
                       engine=engine)
    assert job.get_calibration() == calibration
    engine.get_calibration.assert_called_once_with('hobbes')


def test_get_cancel():
    engine = mock.Mock()
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve')
    job = cg.EngineJob(job_config=cg.JobConfig(job_id='steve'),
                       job=job,
                       engine=engine)
    job.cancel()
    engine.cancel_job.assert_called_once()


def test_results():
    engine = mock.Mock()
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve',
                            execution_status=qtypes.ExecutionStatus(
                                state=qtypes.ExecutionStatus.State.SUCCESS))
    results = mock.Mock()
    engine.get_job_results.return_value = results

    job = cg.EngineJob(job_config=cg.JobConfig(job_id='steve'),
                       job=job,
                       engine=engine)
    assert job.results() == results
    engine.get_job_results.assert_called_once()


@mock.patch('time.sleep', return_value=None)
def test_timeout(patched_time_sleep):
    engine = mock.Mock()
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve',
                            execution_status=qtypes.ExecutionStatus(
                                state=qtypes.ExecutionStatus.State.RUNNING))
    engine.get_job.return_value = job
    job = cg.EngineJob(job_config=cg.JobConfig(job_id='steve'),
                       job=job,
                       engine=engine)
    with pytest.raises(RuntimeError, match='Timed out'):
        job.results()


@mock.patch('cirq.google.engine.client.quantum.QuantumEngineServiceClient',
            autospec=True)
def test_calibration_from_job_with_no_calibration(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client

    client.create_quantum_program.return_value = qtypes.QuantumProgram(
        name='projects/project-id/programs/test')
    client.create_quantum_job.return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={'state': 'SUCCESS'})

    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(program=cirq.Circuit())

    calibration = job.get_calibration()
    assert not calibration
    assert not client.get_quantum_calibration.called


def test_str():
    engine = mock.Mock()
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve')
    job = cg.EngineJob(job_config=cg.JobConfig(job_id='steve'),
                       job=job,
                       engine=engine)
    assert str(job) == 'EngineJob(projects/a/programs/b/jobs/steve)'

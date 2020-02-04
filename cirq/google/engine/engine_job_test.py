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
import numpy as np
import pytest

from google.protobuf import any_pb2
from google.protobuf.text_format import Merge
from google.api_core import exceptions

import cirq
import cirq.google as cg
from cirq.google.api import v1, v2


def _to_any(proto):
    any_proto = qtypes.any_pb2.Any()
    any_proto.Pack(proto)
    return any_proto


def _to_timestamp(json_string):
    timestamp_proto = qtypes.timestamp_pb2.Timestamp()
    timestamp_proto.FromJsonString(json_string)
    return timestamp_proto


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_status(get_job):
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve',
                            execution_status=qtypes.ExecutionStatus(
                                state=qtypes.ExecutionStatus.State.RUNNING))
    get_job.return_value = job

    job = cg.EngineJob(job_id='steve', program=program, job=job)
    assert job.status() == 'RUNNING'
    get_job.assert_called_once()


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_calibration')
@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_get_calibration(get_job, get_calibration):
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = qtypes.QuantumJob(
        name='projects/a/programs/b/jobs/steve',
        execution_status=qtypes.ExecutionStatus(
            calibration_name='projects/a/processors/p/calibrations/123'))
    get_job.return_value = job
    calibration = qtypes.QuantumCalibration(data=_to_any(
        Merge(
            """
    timestamp_ms: 123000,
    metrics: [{
        name: 'xeb',
        targets: ['0_0', '0_1'],
        values: [{
            double_val: .9999
        }]
    }, {
        name: 't1',
        targets: ['q0_0'],
        values: [{
            double_val: 321
        }]
    }, {
        name: 'globalMetric',
        values: [{
            int32_val: 12300
        }]
    }]
""", v2.metrics_pb2.MetricsSnapshot())))
    get_calibration.return_value = calibration

    job = cg.EngineJob(job_id='steve', program=program, job=job)
    assert list(job.get_calibration()) == ['xeb', 't1', 'globalMetric']
    get_calibration.assert_called_once_with('a', 'p', 123)


@mock.patch('cirq.google.engine.engine_client.EngineClient.cancel_job')
def test_cancel(cancel_job):
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve')
    job = cg.EngineJob(job_id='steve', program=program, job=job)
    job.cancel()
    cancel_job.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq.google.engine.engine_client.EngineClient.delete_job')
def test_delete(delete_job):
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve')
    job = cg.EngineJob(job_id='steve', program=program, job=job)
    job.delete()
    delete_job.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job_results')
def test_results(get_job_results):
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve',
                            execution_status=qtypes.ExecutionStatus(
                                state=qtypes.ExecutionStatus.State.SUCCESS))
    results = qtypes.QuantumResult(result=_to_any(
        Merge(
            """
    sweep_results: [{
            repetitions: 4,
            parameterized_results: [{
                params: {
                    assignments: {
                        key: 'a'
                        value: 1
                    }
                },
                measurement_results: {
                    key: 'q'
                    qubit_measurement_results: [{
                      qubit: {
                        id: '1_1'
                      }
                      results: '\006'
                    }]
                }
            },{
                params: {
                    assignments: {
                        key: 'a'
                        value: 2
                    }
                },
                measurement_results: {
                    key: 'q'
                    qubit_measurement_results: [{
                      qubit: {
                        id: '1_1'
                      }
                      results: '\005'
                    }]
                }
            }]
        }]
    """, v2.result_pb2.Result())))
    get_job_results.return_value = results

    job = cg.EngineJob(job_id='steve', program=program, job=job)
    data = job.results()
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'
    get_job_results.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
@mock.patch('time.sleep', return_value=None)
def test_timeout(patched_time_sleep, get_job):
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve',
                            execution_status=qtypes.ExecutionStatus(
                                state=qtypes.ExecutionStatus.State.RUNNING))
    get_job.return_value = job
    job = cg.EngineJob(job_id='steve', program=program, job=job)
    with pytest.raises(RuntimeError, match='Timed out'):
        job.results()


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_calibration')
def test_calibration_from_job_with_no_calibration(get_calibration):
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = cg.EngineJob(job_id='steve',
                       program=program,
                       job=qtypes.QuantumJob(
                           name='projects/project-id/programs/test/jobs/test',
                           execution_status={'state': 'SUCCESS'}))
    calibration = job.get_calibration()
    assert not calibration
    assert not get_calibration.called


def test_str():
    program = cg.EngineProgram('b', cg.Engine('a'), qtypes.QuantumProgram())
    job = qtypes.QuantumJob(name='projects/a/programs/b/jobs/steve')
    job = cg.EngineJob(job_id='steve', program=program, job=job)
    assert str(
        job
    ) == 'EngineJob(project_id=\'a\', program_id=\'b\', job_id=\'steve\')'

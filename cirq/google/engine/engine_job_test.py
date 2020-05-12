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

import datetime
from unittest import mock
import pytest

from google.protobuf.text_format import Merge

import cirq
import cirq.google as cg
from cirq.google.api import v1, v2
from cirq.google.engine.client.quantum_v1alpha1 import types as qtypes
from cirq.google.engine.engine import EngineContext


def _to_any(proto):
    any_proto = qtypes.any_pb2.Any()
    any_proto.Pack(proto)
    return any_proto


@pytest.fixture(scope='session', autouse=True)
def mock_grpc_client():
    with mock.patch('cirq.google.engine.engine_client'
                    '.quantum.QuantumEngineServiceClient') as _fixture:
        yield _fixture


def test_engine():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.engine().project_id == 'a'


def test_program():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.program().project_id == 'a'
    assert job.program().program_id == 'b'


def test_create_time():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=qtypes.QuantumJob(create_time=qtypes.timestamp_pb2.Timestamp(
            seconds=1581515101)))
    assert job.create_time() == datetime.datetime(2020, 2, 12, 13, 45, 1)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_update_time(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = qtypes.QuantumJob(
        update_time=qtypes.timestamp_pb2.Timestamp(seconds=1581515101))
    assert job.update_time() == datetime.datetime(2020, 2, 12, 13, 45, 1)
    get_job.assert_called_once_with('a', 'b', 'steve', False)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_description(get_job):
    job = cg.EngineJob('a',
                       'b',
                       'steve',
                       EngineContext(),
                       _job=qtypes.QuantumJob(description='hello'))
    assert job.description() == 'hello'
    get_job.return_value = qtypes.QuantumJob(description='hello')
    assert cg.EngineJob('a', 'b', 'steve',
                        EngineContext()).description() == 'hello'
    get_job.assert_called_once_with('a', 'b', 'steve', False)


@mock.patch('cirq.google.engine.engine_client.EngineClient.set_job_description')
def test_set_description(set_job_description):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    set_job_description.return_value = qtypes.QuantumJob(description='world')
    assert job.set_description('world').description() == 'world'
    set_job_description.assert_called_with('a', 'b', 'steve', 'world')

    set_job_description.return_value = qtypes.QuantumJob(description='')
    assert job.set_description('').description() == ''
    set_job_description.assert_called_with('a', 'b', 'steve', '')


def test_labels():
    job = cg.EngineJob('a',
                       'b',
                       'steve',
                       EngineContext(),
                       _job=qtypes.QuantumJob(labels={'t': '1'}))
    assert job.labels() == {'t': '1'}


@mock.patch('cirq.google.engine.engine_client.EngineClient.set_job_labels')
def test_set_labels(set_job_labels):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    set_job_labels.return_value = qtypes.QuantumJob(labels={'a': '1', 'b': '1'})
    assert job.set_labels({'a': '1', 'b': '1'}).labels() == {'a': '1', 'b': '1'}
    set_job_labels.assert_called_with('a', 'b', 'steve', {'a': '1', 'b': '1'})

    set_job_labels.return_value = qtypes.QuantumJob()
    assert job.set_labels({}).labels() == {}
    set_job_labels.assert_called_with('a', 'b', 'steve', {})


@mock.patch('cirq.google.engine.engine_client.EngineClient.add_job_labels')
def test_add_labels(add_job_labels):
    job = cg.EngineJob('a',
                       'b',
                       'steve',
                       EngineContext(),
                       _job=qtypes.QuantumJob(labels={}))
    assert job.labels() == {}

    add_job_labels.return_value = qtypes.QuantumJob(labels={
        'a': '1',
    })
    assert job.add_labels({'a': '1'}).labels() == {'a': '1'}
    add_job_labels.assert_called_with('a', 'b', 'steve', {'a': '1'})

    add_job_labels.return_value = qtypes.QuantumJob(labels={'a': '2', 'b': '1'})
    assert job.add_labels({'a': '2', 'b': '1'}).labels() == {'a': '2', 'b': '1'}
    add_job_labels.assert_called_with('a', 'b', 'steve', {'a': '2', 'b': '1'})


@mock.patch('cirq.google.engine.engine_client.EngineClient.remove_job_labels')
def test_remove_labels(remove_job_labels):
    job = cg.EngineJob('a',
                       'b',
                       'steve',
                       EngineContext(),
                       _job=qtypes.QuantumJob(labels={
                           'a': '1',
                           'b': '1'
                       }))
    assert job.labels() == {'a': '1', 'b': '1'}

    remove_job_labels.return_value = qtypes.QuantumJob(labels={
        'b': '1',
    })
    assert job.remove_labels(['a']).labels() == {'b': '1'}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a'])

    remove_job_labels.return_value = qtypes.QuantumJob(labels={})
    assert job.remove_labels(['a', 'b', 'c']).labels() == {}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a', 'b', 'c'])


def test_processor_ids():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=qtypes.QuantumJob(scheduling_config=qtypes.SchedulingConfig(
            processor_selector=qtypes.SchedulingConfig.ProcessorSelector(
                processor_names=['projects/a/processors/p']))))
    assert job.processor_ids() == ['p']


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_status(get_job):
    qjob = qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus(
        state=qtypes.ExecutionStatus.State.RUNNING))
    get_job.return_value = qjob

    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.status() == 'RUNNING'
    get_job.assert_called_once()


def test_failure():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus(
            state=qtypes.ExecutionStatus.State.FAILURE,
            failure=qtypes.ExecutionStatus.Failure(
                error_code=qtypes.ExecutionStatus.Failure.Code.SYSTEM_ERROR,
                error_message='boom'))))
    assert job.failure() == ('SYSTEM_ERROR', 'boom')


def test_failure_with_no_error():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus(
            state=qtypes.ExecutionStatus.State.SUCCESS,)))
    assert not job.failure()


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_get_repetitions_and_sweeps(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = qtypes.QuantumJob(run_context=_to_any(
        v2.run_context_pb2.RunContext(parameter_sweeps=[
            v2.run_context_pb2.ParameterSweep(repetitions=10)
        ])))
    assert job.get_repetitions_and_sweeps() == (10, [cirq.UnitSweep])
    get_job.assert_called_once_with('a', 'b', 'steve', True)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_get_repetitions_and_sweeps_v1(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = qtypes.QuantumJob(run_context=_to_any(
        v1.program_pb2.RunContext(
            parameter_sweeps=[v1.params_pb2.ParameterSweep(repetitions=10)])))
    with pytest.raises(ValueError, match='v1 RunContext is not supported'):
        job.get_repetitions_and_sweeps()


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
def test_get_repetitions_and_sweeps_unsupported(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = qtypes.QuantumJob(run_context=qtypes.any_pb2.Any(
        type_url='type.googleapis.com/unknown.proto'))
    with pytest.raises(ValueError,
                       match='unsupported run_context type: unknown.proto'):
        job.get_repetitions_and_sweeps()


def test_get_processor():
    qjob = qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus(
        processor_name='projects/a/processors/p'))

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert job.get_processor().processor_id == 'p'


def test_get_processor_no_processor():
    qjob = qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus())

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert not job.get_processor()


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_calibration')
def test_get_calibration(get_calibration):
    qjob = qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus(
        calibration_name='projects/a/processors/p/calibrations/123'))
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

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert list(job.get_calibration()) == ['xeb', 't1', 'globalMetric']
    get_calibration.assert_called_once_with('a', 'p', 123)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_calibration')
def test_calibration__with_no_calibration(get_calibration):
    job = cg.EngineJob('a',
                       'b',
                       'steve',
                       EngineContext(),
                       _job=qtypes.QuantumJob(
                           name='projects/project-id/programs/test/jobs/test',
                           execution_status={'state': 'SUCCESS'}))
    calibration = job.get_calibration()
    assert not calibration
    assert not get_calibration.called


@mock.patch('cirq.google.engine.engine_client.EngineClient.cancel_job')
def test_cancel(cancel_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    job.cancel()
    cancel_job.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq.google.engine.engine_client.EngineClient.delete_job')
def test_delete(delete_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    job.delete()
    delete_job.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job_results')
def test_results(get_job_results):
    qjob = qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus(
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

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    data = job.results()
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'
    get_job_results.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job')
@mock.patch('time.sleep', return_value=None)
def test_timeout(patched_time_sleep, get_job):
    qjob = qtypes.QuantumJob(execution_status=qtypes.ExecutionStatus(
        state=qtypes.ExecutionStatus.State.RUNNING))
    get_job.return_value = qjob
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(timeout=500))
    with pytest.raises(RuntimeError, match='Timed out'):
        job.results()


def test_str():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert str(
        job
    ) == 'EngineJob(project_id=\'a\', program_id=\'b\', job_id=\'steve\')'

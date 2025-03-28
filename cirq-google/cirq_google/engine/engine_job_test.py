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

import duet
import pytest
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge

import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.cloud import quantum
from cirq_google.engine import util
from cirq_google.engine.engine import EngineContext


@pytest.fixture(scope='module', autouse=True)
def mock_grpc_client():
    with mock.patch(
        'cirq_google.engine.engine_client.quantum.QuantumEngineServiceClient'
    ) as _fixture:
        yield _fixture


def test_engine():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.engine().project_id == 'a'


def test_program():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.program().project_id == 'a'
    assert job.program().program_id == 'b'


def test_id():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(create_time=timestamp_pb2.Timestamp(seconds=1581515101)),
    )
    assert job.id() == 'steve'


def test_create_time():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(create_time=timestamp_pb2.Timestamp(seconds=1581515101)),
    )
    assert job.create_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_update_time(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(
        update_time=timestamp_pb2.Timestamp(seconds=1581515101)
    )
    assert job.update_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )
    get_job.assert_called_once_with('a', 'b', 'steve', False)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_description(get_job):
    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(description='hello')
    )
    assert job.description() == 'hello'
    get_job.return_value = quantum.QuantumJob(description='hello')
    assert cg.EngineJob('a', 'b', 'steve', EngineContext()).description() == 'hello'
    get_job.assert_called_once_with('a', 'b', 'steve', False)


@mock.patch('cirq_google.engine.engine_client.EngineClient.set_job_description_async')
def test_set_description(set_job_description):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    set_job_description.return_value = quantum.QuantumJob(description='world')
    assert job.set_description('world').description() == 'world'
    set_job_description.assert_called_with('a', 'b', 'steve', 'world')

    set_job_description.return_value = quantum.QuantumJob(description='')
    assert job.set_description('').description() == ''
    set_job_description.assert_called_with('a', 'b', 'steve', '')


def test_labels():
    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(labels={'t': '1'})
    )
    assert job.labels() == {'t': '1'}


@mock.patch('cirq_google.engine.engine_client.EngineClient.set_job_labels_async')
def test_set_labels(set_job_labels):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    set_job_labels.return_value = quantum.QuantumJob(labels={'a': '1', 'b': '1'})
    assert job.set_labels({'a': '1', 'b': '1'}).labels() == {'a': '1', 'b': '1'}
    set_job_labels.assert_called_with('a', 'b', 'steve', {'a': '1', 'b': '1'})

    set_job_labels.return_value = quantum.QuantumJob()
    assert job.set_labels({}).labels() == {}
    set_job_labels.assert_called_with('a', 'b', 'steve', {})


@mock.patch('cirq_google.engine.engine_client.EngineClient.add_job_labels_async')
def test_add_labels(add_job_labels):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(labels={}))
    assert job.labels() == {}

    add_job_labels.return_value = quantum.QuantumJob(labels={'a': '1'})
    assert job.add_labels({'a': '1'}).labels() == {'a': '1'}
    add_job_labels.assert_called_with('a', 'b', 'steve', {'a': '1'})

    add_job_labels.return_value = quantum.QuantumJob(labels={'a': '2', 'b': '1'})
    assert job.add_labels({'a': '2', 'b': '1'}).labels() == {'a': '2', 'b': '1'}
    add_job_labels.assert_called_with('a', 'b', 'steve', {'a': '2', 'b': '1'})


@mock.patch('cirq_google.engine.engine_client.EngineClient.remove_job_labels_async')
def test_remove_labels(remove_job_labels):
    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(labels={'a': '1', 'b': '1'})
    )
    assert job.labels() == {'a': '1', 'b': '1'}

    remove_job_labels.return_value = quantum.QuantumJob(labels={'b': '1'})
    assert job.remove_labels(['a']).labels() == {'b': '1'}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a'])

    remove_job_labels.return_value = quantum.QuantumJob(labels={})
    assert job.remove_labels(['a', 'b', 'c']).labels() == {}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a', 'b', 'c'])


def test_processor_ids():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(
            scheduling_config=quantum.SchedulingConfig(
                processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                    processor_names=['projects/a/processors/p']
                )
            )
        ),
    )
    assert job.processor_ids() == ['p']


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_status(get_job):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.RUNNING)
    )
    get_job.return_value = qjob

    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.status() == 'RUNNING'
    get_job.assert_called_once()
    assert job.execution_status() == quantum.ExecutionStatus.State.RUNNING


def test_failure():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(
            execution_status=quantum.ExecutionStatus(
                state=quantum.ExecutionStatus.State.FAILURE,
                failure=quantum.ExecutionStatus.Failure(
                    error_code=quantum.ExecutionStatus.Failure.Code.SYSTEM_ERROR,
                    error_message='boom',
                ),
            )
        ),
    )
    assert job.failure() == ('SYSTEM_ERROR', 'boom')


def test_failure_with_no_error():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(
            execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS)
        ),
    )
    assert not job.failure()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=10)]
            )
        )
    )
    assert job.get_repetitions_and_sweeps() == (10, [cirq.UnitSweep])
    get_job.assert_called_once_with('a', 'b', 'steve', True)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps_v1(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(
            v1.program_pb2.RunContext(
                parameter_sweeps=[v1.params_pb2.ParameterSweep(repetitions=10)]
            )
        )
    )
    with pytest.raises(ValueError, match='v1 RunContext is not supported'):
        job.get_repetitions_and_sweeps()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps_unsupported(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(
        run_context=any_pb2.Any(type_url='type.googleapis.com/unknown.proto')
    )
    with pytest.raises(ValueError, match='unsupported run_context type: unknown.proto'):
        job.get_repetitions_and_sweeps()


def test_get_processor():
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(processor_name='projects/a/processors/p')
    )

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert job.get_processor().processor_id == 'p'


def test_get_processor_no_processor():
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus())

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert not job.get_processor()


@mock.patch('cirq_google.engine.engine_client.EngineClient.cancel_job_async')
def test_cancel(cancel_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    job.cancel()
    cancel_job.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_job_async')
def test_delete(delete_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    job.delete()
    delete_job.assert_called_once_with('a', 'b', 'steve')


RESULTS = quantum.QuantumResult(
    result=util.pack_any(
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
""",
            v2.result_pb2.Result(),
        )
    )
)


UPDATE_TIME = datetime.datetime.now(tz=datetime.timezone.utc)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    data = job.results()
    assert len(data) == 2
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'
    get_job_results.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_iter(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    results = [str(r) for r in job]
    assert len(results) == 2
    assert results[0] == 'q=0110'
    assert results[1] == 'q=1010'


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_getitem(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert str(job[0]) == 'q=0110'
    assert str(job[1]) == 'q=1010'
    with pytest.raises(IndexError):
        _ = job[2]


def test_receives_results_via_stream_returns_correct_results():
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    result_future = duet.completed_future(RESULTS)

    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=qjob, job_result_future=result_future
    )
    data = job.results()

    assert len(data) == 2
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'


def test_receives_job_via_stream_raises_and_updates_underlying_job():
    expected_error_code = quantum.ExecutionStatus.Failure.Code.SYSTEM_ERROR
    expected_error_message = 'system error'
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(
            state=quantum.ExecutionStatus.State.SUCCESS,
            failure=quantum.ExecutionStatus.Failure(
                error_code=expected_error_code, error_message=expected_error_message
            ),
        ),
        update_time=UPDATE_TIME,
    )
    result_future = duet.completed_future(qjob)

    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=qjob, job_result_future=result_future
    )
    qjob.execution_status.state = quantum.ExecutionStatus.State.FAILURE

    with pytest.raises(RuntimeError):
        job.results()
    actual_error_code, actual_error_message = job.failure()

    # Checks that the underlying job has been updated by checking failure information.
    assert actual_error_code == expected_error_code.name
    assert actual_error_message == expected_error_message


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_len(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert len(job) == 2


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_timeout(get_job):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.RUNNING),
        update_time=UPDATE_TIME,
    )
    get_job.return_value = qjob
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(timeout=0.1))
    with pytest.raises(TimeoutError):
        job.results()


def test_str():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert str(job) == 'EngineJob(project_id=\'a\', program_id=\'b\', job_id=\'steve\')'

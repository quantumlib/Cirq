# Copyright 2020 The Cirq Developers
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
"""Tests for EngineClient."""
import datetime
from unittest import mock
import pytest
from google.api_core import exceptions
from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine.engine_client import EngineClient, EngineException
from cirq_google.engine.client import quantum
from cirq_google.engine.client.quantum_v1alpha1 import enums as qenums
from cirq_google.engine.client.quantum_v1alpha1 import types as qtypes


def setup_mock_(client_constructor):
    grpc_client = mock.Mock()
    client_constructor.return_value = grpc_client
    return grpc_client


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_create_program(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.create_quantum_program.return_value = result

    code = qtypes.any_pb2.Any()
    labels = {'hello': 'world'}
    client = EngineClient()
    assert client.create_program('proj', 'prog', code, 'A program', labels) == ('prog', result)
    assert grpc_client.create_quantum_program.call_args[0] == (
        'projects/proj',
        qtypes.QuantumProgram(
            name='projects/proj/programs/prog', code=code, description='A program', labels=labels
        ),
        False,
    )

    assert client.create_program('proj', 'prog', code, 'A program') == ('prog', result)
    assert grpc_client.create_quantum_program.call_args[0] == (
        'projects/proj',
        qtypes.QuantumProgram(
            name='projects/proj/programs/prog', code=code, description='A program'
        ),
        False,
    )

    assert client.create_program('proj', 'prog', code, labels=labels) == ('prog', result)
    assert grpc_client.create_quantum_program.call_args[0] == (
        'projects/proj',
        qtypes.QuantumProgram(name='projects/proj/programs/prog', code=code, labels=labels),
        False,
    )

    assert client.create_program('proj', 'prog', code) == ('prog', result)
    assert grpc_client.create_quantum_program.call_args[0] == (
        'projects/proj',
        qtypes.QuantumProgram(name='projects/proj/programs/prog', code=code),
        False,
    )

    assert client.create_program('proj', program_id=None, code=code) == ('prog', result)
    assert grpc_client.create_quantum_program.call_args[0] == (
        'projects/proj',
        qtypes.QuantumProgram(code=code),
        False,
    )


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_program(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.get_quantum_program.return_value = result

    client = EngineClient()
    assert client.get_program('proj', 'prog', False) == result
    assert grpc_client.get_quantum_program.call_args[0] == ('projects/proj/programs/prog', False)

    assert client.get_program('proj', 'prog', True) == result
    assert grpc_client.get_quantum_program.call_args[0] == ('projects/proj/programs/prog', True)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_program(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    results = [
        qtypes.QuantumProgram(name='projects/proj/programs/prog1'),
        qtypes.QuantumProgram(name='projects/proj/programs/prog2'),
    ]
    grpc_client.list_quantum_programs.return_value = results

    client = EngineClient()
    assert client.list_programs(project_id='proj') == results
    assert grpc_client.list_quantum_programs.call_args[0] == ('projects/proj',)
    assert grpc_client.list_quantum_programs.call_args[1] == {
        'filter_': '',
    }


# yapf: disable
@pytest.mark.parametrize(
    'expected_filter, created_after, created_before, labels',
    [
        ('',
            None,
            None,
            None),
        ('create_time >= 2020-09-01',
            datetime.date(2020, 9, 1),
            None,
            None),
        ('create_time >= 1598918400',
            datetime.datetime(2020, 9, 1, 0, 0, 0,
                              tzinfo=datetime.timezone.utc),
            None,
            None),
        ('create_time <= 2020-10-01',
            None,
            datetime.date(2020, 10, 1),
            None),
        ('create_time >= 2020-09-01 AND create_time <= 1598918410',
            datetime.date(2020, 9, 1),
            datetime.datetime(2020, 9, 1, 0, 0, 10,
                            tzinfo=datetime.timezone.utc),
            None),
        ('labels.color:red AND labels.shape:*',
            None,
            None,
            {
            'color': 'red',
            'shape': '*'
            },
        ),
        ('create_time >= 2020-08-01 AND '
         'create_time <= 1598918400 AND '
         'labels.color:red AND labels.shape:*',
            datetime.date(2020, 8, 1),
            datetime.datetime(2020, 9, 1, tzinfo=datetime.timezone.utc),
            {
            'color': 'red',
            'shape': '*'
            },
        ),
    ])
# yapf: enable
@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_program_filters(client_constructor, expected_filter,
                              created_before, created_after, labels):
    grpc_client = setup_mock_(client_constructor)
    client = EngineClient()
    client.list_programs(project_id='proj',
                         created_before=created_before,
                         created_after=created_after,
                         has_labels=labels)
    assert grpc_client.list_quantum_programs.call_args[1] == {
        'filter_': expected_filter,
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_program_filters_invalid_type(client_constructor):
    with pytest.raises(ValueError, match=""):
        EngineClient().list_programs(project_id='proj',
                                     created_before="Unsupported date/time")


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_set_program_description(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    assert client.set_program_description('proj', 'prog', 'A program') == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog',
                              description='A program'),
        qtypes.field_mask_pb2.FieldMask(paths=['description']))

    assert client.set_program_description('proj', 'prog', '') == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog'),
        qtypes.field_mask_pb2.FieldMask(paths=['description']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_set_program_labels(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    grpc_client.get_quantum_program.return_value = qtypes.QuantumProgram(
        labels={
            'color': 'red',
            'weather': 'sun',
            'run': '1'
        },
        label_fingerprint='hash')
    result = qtypes.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    labels = {'hello': 'world', 'color': 'blue', 'run': '1'}
    assert client.set_program_labels('proj', 'prog', labels) == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog',
                              labels=labels,
                              label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))

    assert client.set_program_labels('proj', 'prog', {}) == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog',
                              label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_add_program_labels(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    existing = qtypes.QuantumProgram(labels={
        'color': 'red',
        'weather': 'sun',
        'run': '1'
    },
                                     label_fingerprint='hash')
    grpc_client.get_quantum_program.return_value = existing
    result = qtypes.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    assert client.add_program_labels('proj', 'prog',
                                     {'color': 'red'}) == existing
    assert grpc_client.update_quantum_program.call_count == 0

    assert client.add_program_labels('proj', 'prog',
                                     {'hello': 'world'}) == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog',
                              labels={
                                  'color': 'red',
                                  'weather': 'sun',
                                  'run': '1',
                                  'hello': 'world'
                              },
                              label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))

    assert client.add_program_labels('proj', 'prog', {
        'hello': 'world',
        'color': 'blue'
    }) == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog',
                              labels={
                                  'color': 'blue',
                                  'weather': 'sun',
                                  'run': '1',
                                  'hello': 'world'
                              },
                              label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_remove_program_labels(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    existing = qtypes.QuantumProgram(labels={
        'color': 'red',
        'weather': 'sun',
        'run': '1'
    },
                                     label_fingerprint='hash')
    grpc_client.get_quantum_program.return_value = existing
    result = qtypes.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    assert client.remove_program_labels('proj', 'prog', ['other']) == existing
    assert grpc_client.update_quantum_program.call_count == 0

    assert client.remove_program_labels('proj', 'prog',
                                        ['hello', 'weather']) == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog',
                              labels={
                                  'color': 'red',
                                  'run': '1',
                              },
                              label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))

    assert client.remove_program_labels('proj', 'prog',
                                        ['color', 'weather', 'run']) == result
    assert grpc_client.update_quantum_program.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumProgram(name='projects/proj/programs/prog',
                              label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_delete_program(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    client = EngineClient()
    assert not client.delete_program('proj', 'prog')
    assert grpc_client.delete_quantum_program.call_args[0] == (
        'projects/proj/programs/prog', False)

    assert not client.delete_program('proj', 'prog', delete_jobs=True)
    assert grpc_client.delete_quantum_program.call_args[0] == (
        'projects/proj/programs/prog', True)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_create_job(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.create_quantum_job.return_value = result

    run_context = qtypes.any_pb2.Any()
    labels = {'hello': 'world'}
    client = EngineClient()
    assert client.create_job('proj', 'prog', 'job0', ['processor0'],
                             run_context, 10, 'A job',
                             labels) == ('job0', result)
    assert grpc_client.create_quantum_job.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumJob(
            name='projects/proj/programs/prog/jobs/job0',
            run_context=run_context,
            scheduling_config=qtypes.SchedulingConfig(
                priority=10,
                processor_selector=qtypes.SchedulingConfig.ProcessorSelector(
                    processor_names=['projects/proj/processors/processor0'])),
            description='A job',
            labels=labels), False)

    assert client.create_job(
        'proj',
        'prog',
        'job0',
        ['processor0'],
        run_context,
        10,
        'A job',
    ) == ('job0', result)
    assert grpc_client.create_quantum_job.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumJob(
            name='projects/proj/programs/prog/jobs/job0',
            run_context=run_context,
            scheduling_config=qtypes.SchedulingConfig(
                priority=10,
                processor_selector=qtypes.SchedulingConfig.ProcessorSelector(
                    processor_names=['projects/proj/processors/processor0'])),
            description='A job'), False)

    assert client.create_job('proj',
                             'prog',
                             'job0', ['processor0'],
                             run_context,
                             10,
                             labels=labels) == ('job0', result)
    assert grpc_client.create_quantum_job.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumJob(
            name='projects/proj/programs/prog/jobs/job0',
            run_context=run_context,
            scheduling_config=qtypes.SchedulingConfig(
                priority=10,
                processor_selector=qtypes.SchedulingConfig.ProcessorSelector(
                    processor_names=['projects/proj/processors/processor0'])),
            labels=labels), False)

    assert client.create_job('proj', 'prog', 'job0', ['processor0'],
                             run_context, 10) == ('job0', result)
    assert grpc_client.create_quantum_job.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumJob(
            name='projects/proj/programs/prog/jobs/job0',
            run_context=run_context,
            scheduling_config=qtypes.SchedulingConfig(
                priority=10,
                processor_selector=qtypes.SchedulingConfig.ProcessorSelector(
                    processor_names=['projects/proj/processors/processor0'])),
        ), False)

    assert client.create_job('proj',
                             'prog',
                             job_id=None,
                             processor_ids=['processor0'],
                             run_context=run_context,
                             priority=10) == ('job0', result)
    assert grpc_client.create_quantum_job.call_args[0] == (
        'projects/proj/programs/prog',
        qtypes.QuantumJob(
            run_context=run_context,
            scheduling_config=qtypes.SchedulingConfig(
                priority=10,
                processor_selector=qtypes.SchedulingConfig.ProcessorSelector(
                    processor_names=['projects/proj/processors/processor0'])),
        ), False)

    with pytest.raises(ValueError, match='priority must be between 0 and 1000'):
        client.create_job('proj',
                          'prog',
                          job_id=None,
                          processor_ids=['processor0'],
                          run_context=run_context,
                          priority=5000)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_job(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.get_quantum_job.return_value = result

    client = EngineClient()
    assert client.get_job('proj', 'prog', 'job0', False) == result
    assert grpc_client.get_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0', False)

    assert client.get_job('proj', 'prog', 'job0', True) == result
    assert grpc_client.get_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0', True)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_set_job_description(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    assert client.set_job_description('proj', 'prog', 'job0', 'A job') == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0',
                          description='A job'),
        qtypes.field_mask_pb2.FieldMask(paths=['description']))

    assert client.set_job_description('proj', 'prog', 'job0', '') == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0'),
        qtypes.field_mask_pb2.FieldMask(paths=['description']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_set_job_labels(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    grpc_client.get_quantum_job.return_value = qtypes.QuantumJob(
        labels={
            'color': 'red',
            'weather': 'sun',
            'run': '1'
        },
        label_fingerprint='hash')
    result = qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    labels = {'hello': 'world', 'color': 'blue', 'run': '1'}
    assert client.set_job_labels('proj', 'prog', 'job0', labels) == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0',
                          labels=labels,
                          label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))

    assert client.set_job_labels('proj', 'prog', 'job0', {}) == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0',
                          label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_add_job_labels(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    existing = qtypes.QuantumJob(labels={
        'color': 'red',
        'weather': 'sun',
        'run': '1'
    },
                                 label_fingerprint='hash')
    grpc_client.get_quantum_job.return_value = existing
    result = qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    assert client.add_job_labels('proj', 'prog', 'job0',
                                 {'color': 'red'}) == existing
    assert grpc_client.update_quantum_job.call_count == 0

    assert client.add_job_labels('proj', 'prog', 'job0',
                                 {'hello': 'world'}) == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0',
                          labels={
                              'color': 'red',
                              'weather': 'sun',
                              'run': '1',
                              'hello': 'world'
                          },
                          label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))

    assert client.add_job_labels('proj', 'prog', 'job0', {
        'hello': 'world',
        'color': 'blue'
    }) == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0',
                          labels={
                              'color': 'blue',
                              'weather': 'sun',
                              'run': '1',
                              'hello': 'world'
                          },
                          label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_remove_job_labels(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    existing = qtypes.QuantumJob(labels={
        'color': 'red',
        'weather': 'sun',
        'run': '1'
    },
                                 label_fingerprint='hash')
    grpc_client.get_quantum_job.return_value = existing
    result = qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    assert client.remove_job_labels('proj', 'prog', 'job0',
                                    ['other']) == existing
    assert grpc_client.update_quantum_program.call_count == 0

    assert client.remove_job_labels('proj', 'prog', 'job0',
                                    ['hello', 'weather']) == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0',
                          labels={
                              'color': 'red',
                              'run': '1',
                          },
                          label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))

    assert client.remove_job_labels('proj', 'prog', 'job0',
                                    ['color', 'weather', 'run']) == result
    assert grpc_client.update_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job0',
                          label_fingerprint='hash'),
        qtypes.field_mask_pb2.FieldMask(paths=['labels']))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_delete_job(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    client = EngineClient()
    assert not client.delete_job('proj', 'prog', 'job0')
    assert grpc_client.delete_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_cancel_job(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    client = EngineClient()
    assert not client.cancel_job('proj', 'prog', 'job0')
    assert grpc_client.cancel_quantum_job.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_job_results(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumResult(
        parent='projects/proj/programs/prog/jobs/job0')
    grpc_client.get_quantum_result.return_value = result

    client = EngineClient()
    assert client.get_job_results('proj', 'prog', 'job0') == result
    assert grpc_client.get_quantum_result.call_args[0] == (
        'projects/proj/programs/prog/jobs/job0',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_jobs(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    results = [
        qtypes.QuantumJob(name='projects/proj/programs/prog1/jobs/job1'),
        qtypes.QuantumJob(name='projects/proj/programs/prog1/jobs/job2')
    ]
    grpc_client.list_quantum_jobs.return_value = results

    client = EngineClient()
    assert client.list_jobs(project_id='proj', program_id='prog1') == results
    assert grpc_client.list_quantum_jobs.call_args[0] == (
        'projects/proj/programs/prog1',)
    assert grpc_client.list_quantum_jobs.call_args[1] == {
        'filter_': '',
    }

    assert client.list_jobs(project_id='proj') == results
    assert grpc_client.list_quantum_jobs.call_args[0] == (
        'projects/proj/programs/-',)
    assert grpc_client.list_quantum_jobs.call_args[1] == {
        'filter_': '',
    }


# yapf: disable
@pytest.mark.parametrize(
    'expected_filter, '
        'created_after, '
        'created_before, '
        'labels, '
        'execution_states, '
        'executed_processor_ids, '
        'scheduled_processor_ids, ',
    [
        ('',
            None,
            None,
            None,
            None,
            None,
            None),
        ('create_time >= 2020-09-01',
            datetime.date(2020, 9, 1),
            None,
            None,
            None,
            None,
            None),
        ('create_time >= 1598918400',
            datetime.datetime(2020, 9, 1, 0, 0, 0,
                              tzinfo=datetime.timezone.utc),
            None,
            None,
            None,
            None,
            None),
        ('create_time <= 2020-10-01',
            None,
            datetime.date(2020, 10, 1),
            None,
            None,
            None,
            None),
        ('create_time >= 2020-09-01 AND create_time <= 1598918410',
            datetime.date(2020, 9, 1),
            datetime.datetime(2020, 9, 1, 0, 0, 10,
                            tzinfo=datetime.timezone.utc),
            None,
            None,
            None,
            None),
        ('labels.color:red AND labels.shape:*',
            None,
            None,
            {
            'color': 'red',
            'shape': '*'
            },
            None,
            None,
            None),
        ('(execution_status.state = FAILURE OR '
         'execution_status.state = CANCELLED)',
            None,
            None,
            None,
            [quantum.enums.ExecutionStatus.State.FAILURE,
             quantum.enums.ExecutionStatus.State.CANCELLED,],
            None,
            None),
        ('create_time >= 2020-08-01 AND '
         'create_time <= 1598918400 AND '
         'labels.color:red AND labels.shape:* AND '
         '(execution_status.state = SUCCESS)',
            datetime.date(2020, 8, 1),
            datetime.datetime(2020, 9, 1, tzinfo=datetime.timezone.utc),
            {
            'color': 'red',
            'shape': '*'
            },
            [quantum.enums.ExecutionStatus.State.SUCCESS,],
            None,
            None),
        ('(executed_processor_id = proc1)',
            None,
            None,
            None,
            None,
            ['proc1'],
            None),
        ('(executed_processor_id = proc1 OR executed_processor_id = proc2)',
            None,
            None,
            None,
            None,
            ['proc1', 'proc2'],
            None),
        ('(scheduled_processor_ids: proc1)',
            None,
            None,
            None,
            None,
            None,
            ['proc1']),
        ('(scheduled_processor_ids: proc1 OR scheduled_processor_ids: proc2)',
            None,
            None,
            None,
            None,
            None,
            ['proc1', 'proc2']),
    ])
# yapf: enable
@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_jobs_filters(client_constructor, expected_filter, created_before,
                           created_after, labels, execution_states,
                           executed_processor_ids, scheduled_processor_ids):
    grpc_client = setup_mock_(client_constructor)
    client = EngineClient()
    client.list_jobs(project_id='proj',
                     program_id='prog',
                     created_before=created_before,
                     created_after=created_after,
                     has_labels=labels,
                     execution_states=execution_states,
                     executed_processor_ids=executed_processor_ids,
                     scheduled_processor_ids=scheduled_processor_ids)
    assert grpc_client.list_quantum_jobs.call_args[1] == {
        'filter_': expected_filter,
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_processors(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    results = [
        qtypes.QuantumProcessor(name='projects/proj/processor/processor0'),
        qtypes.QuantumProcessor(name='projects/proj/processor/processor1')
    ]
    grpc_client.list_quantum_processors.return_value = results

    client = EngineClient()
    assert client.list_processors('proj') == results
    assert grpc_client.list_quantum_processors.call_args[0] == (
        'projects/proj',)
    assert grpc_client.list_quantum_processors.call_args[1] == {
        'filter_': '',
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_processor(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumProcessor(name='projects/proj/processors/processor0')
    grpc_client.get_quantum_processor.return_value = result

    client = EngineClient()
    assert client.get_processor('proj', 'processor0') == result
    assert grpc_client.get_quantum_processor.call_args[0] == (
        'projects/proj/processors/processor0',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_calibrations(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    results = [
        qtypes.QuantumCalibration(
            name='projects/proj/processor/processor0/calibrations/123456'),
        qtypes.QuantumCalibration(
            name='projects/proj/processor/processor1/calibrations/224466')
    ]
    grpc_client.list_quantum_calibrations.return_value = results

    client = EngineClient()
    assert client.list_calibrations('proj', 'processor0') == results
    assert grpc_client.list_quantum_calibrations.call_args[0] == (
        'projects/proj/processors/processor0',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_calibration(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumCalibration(
        name='projects/proj/processors/processor0/calibrations/123456')
    grpc_client.get_quantum_calibration.return_value = result

    client = EngineClient()
    assert client.get_calibration('proj', 'processor0', 123456) == result
    assert grpc_client.get_quantum_calibration.call_args[0] == (
        'projects/proj/processors/processor0/calibrations/123456',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_current_calibration(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    result = qtypes.QuantumCalibration(
        name='projects/proj/processors/processor0/calibrations/123456')
    grpc_client.get_quantum_calibration.return_value = result

    client = EngineClient()
    assert client.get_current_calibration('proj', 'processor0') == result
    assert grpc_client.get_quantum_calibration.call_args[0] == (
        'projects/proj/processors/processor0/calibrations/current',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_current_calibration_does_not_exist(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    grpc_client.get_quantum_calibration.side_effect = exceptions.NotFound(
        'not found')

    client = EngineClient()
    assert client.get_current_calibration('proj', 'processor0') is None
    assert grpc_client.get_quantum_calibration.call_args[0] == (
        'projects/proj/processors/processor0/calibrations/current',)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_current_calibration_error(client_constructor):
    grpc_client = setup_mock_(client_constructor)

    grpc_client.get_quantum_calibration.side_effect = exceptions.BadRequest(
        'boom')

    client = EngineClient()
    with pytest.raises(EngineException, match='boom'):
        client.get_current_calibration('proj', 'processor0')


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_api_doesnt_retry_not_found_errors(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.NotFound(
        'not found')

    client = EngineClient()
    with pytest.raises(EngineException, match='not found'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 1


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_api_retry_5xx_errors(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.ServiceUnavailable(
        'internal error')

    client = EngineClient(max_retry_delay_seconds=0.3)
    with pytest.raises(TimeoutError,
                       match='Reached max retry attempts.*internal error'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 3


@mock.patch('time.sleep', return_value=None)
@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_api_retry_times(client_constructor, mock_time):
    grpc_client = setup_mock_(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.ServiceUnavailable('internal error')

    client = EngineClient(max_retry_delay_seconds=0.3)
    with pytest.raises(TimeoutError, match='Reached max retry attempts.*internal error'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 3

    assert len(mock_time.call_args_list) == 2
    assert all(x == y for (x, _), y in zip(mock_time.call_args_list, [(0.1,), (0.2,)]))


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_create_reservation(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    start = datetime.datetime.fromtimestamp(1000000000)
    end = datetime.datetime.fromtimestamp(1000003600)
    users = ['jeff@google.com']
    result = qtypes.QuantumReservation(
        name='projects/proj/processors/processor0/reservations/papar-party-44',
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=users,
    )
    grpc_client.create_quantum_reservation.return_value = result

    client = EngineClient()
    assert client.create_reservation('proj', 'processor0', start, end,
                                     users) == result
    assert grpc_client.create_quantum_reservation.call_count == 1
    kwargs = grpc_client.create_quantum_reservation.call_args[1]
    # The outgoing argument will not have the resource name
    result.name = ''
    assert kwargs == {
        'parent': 'projects/proj/processors/processor0',
        'quantum_reservation': result
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_cancel_reservation(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.cancel_quantum_reservation.return_value = result

    client = EngineClient()
    assert (client.cancel_reservation('proj', 'processor0',
                                      'papar-party-44') == result)
    kwargs = grpc_client.cancel_quantum_reservation.call_args[1]
    assert kwargs == {
        'name': name,
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_delete_reservation(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.delete_quantum_reservation.return_value = result

    client = EngineClient()
    assert (client.delete_reservation('proj', 'processor0',
                                      'papar-party-44') == result)
    kwargs = grpc_client.delete_quantum_reservation.call_args[1]
    assert kwargs == {
        'name': name,
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_reservation(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.get_quantum_reservation.return_value = result

    client = EngineClient()
    assert (client.get_reservation('proj', 'processor0',
                                   'papar-party-44') == result)
    kwargs = grpc_client.get_quantum_reservation.call_args[1]
    assert kwargs == {
        'name': name,
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_reservation_not_found(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    grpc_client.get_quantum_reservation.side_effect = exceptions.NotFound(
        'not found')

    client = EngineClient()
    assert (client.get_reservation('proj', 'processor0',
                                   'papar-party-44') is None)
    kwargs = grpc_client.get_quantum_reservation.call_args[1]
    assert kwargs == {
        'name': name,
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_reservation_exception(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    grpc_client.get_quantum_reservation.side_effect = exceptions.BadRequest(
        'boom')

    client = EngineClient()
    with pytest.raises(EngineException, match='boom'):
        client.get_reservation('proj', 'processor0', 'goog')


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_reservation(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    results = [
        qtypes.QuantumReservation(
            name=name,
            start_time=Timestamp(seconds=1000000000),
            end_time=Timestamp(seconds=1000002000),
            whitelisted_users=['jeff@google.com'],
        ),
        qtypes.QuantumReservation(
            name=name,
            start_time=Timestamp(seconds=1200000000),
            end_time=Timestamp(seconds=1200002000),
            whitelisted_users=['dstrain@google.com'],
        ),
    ]
    grpc_client.list_quantum_reservations.return_value = results

    client = EngineClient()
    assert (client.list_reservations('proj', 'processor0') == results)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_update_reservation(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000001000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.update_quantum_reservation.return_value = result

    client = EngineClient()
    assert (client.update_reservation(
        'proj',
        'processor0',
        'papar-party-44',
        start=datetime.datetime.fromtimestamp(1000001000),
        end=datetime.datetime.fromtimestamp(1000002000),
        whitelisted_users=['jeff@google.com'],
    ) == result)
    kwargs = grpc_client.update_quantum_reservation.call_args[1]
    assert kwargs == {
        'name':
        name,
        'quantum_reservation':
        result,
        'update_mask':
        FieldMask(paths=['start_time', 'end_time', 'whitelisted_users'])
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_update_reservation_remove_all_users(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = qtypes.QuantumReservation(
        name=name,
        whitelisted_users=[],
    )
    grpc_client.update_quantum_reservation.return_value = result

    client = EngineClient()
    assert (client.update_reservation(
        'proj',
        'processor0',
        'papar-party-44',
        whitelisted_users=[],
    ) == result)
    kwargs = grpc_client.update_quantum_reservation.call_args[1]
    assert kwargs == {
        'name': name,
        'quantum_reservation': result,
        'update_mask': FieldMask(paths=['whitelisted_users'])
    }


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_time_slots(client_constructor):
    grpc_client = setup_mock_(client_constructor)
    results = [
        qtypes.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000020000),
            end_time=Timestamp(seconds=1000040000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            maintenance_config=qtypes.QuantumTimeSlot.MaintenanceConfig(
                title='Testing',
                description='Testing some new configuration.',
            ),
        ),
        qtypes.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000010000),
            end_time=Timestamp(seconds=1000020000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
            reservation_config=qtypes.QuantumTimeSlot.ReservationConfig(
                project_id='super_secret_quantum'),
        )
    ]
    grpc_client.list_quantum_time_slots.return_value = results

    client = EngineClient()
    assert (client.list_time_slots('proj', 'processor0') == results)

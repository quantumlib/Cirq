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
import asyncio
import datetime
import os
from unittest import mock

import duet
import pytest
from google.api_core import exceptions
from google.protobuf import any_pb2
from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.timestamp_pb2 import Timestamp

from cirq_google.engine.engine_client import EngineClient, EngineException
import cirq_google.engine.stream_manager as engine_stream_manager
from cirq_google.cloud import quantum


def _setup_client_mock(client_constructor):
    grpc_client = mock.AsyncMock()
    client_constructor.return_value = grpc_client
    return grpc_client


def _setup_stream_manager_mock(manager_constructor):
    stream_manager = mock.MagicMock()
    manager_constructor.return_value = stream_manager
    return stream_manager


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_create_program(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.create_quantum_program.return_value = result

    code = any_pb2.Any()
    labels = {'hello': 'world'}
    client = EngineClient()

    assert client.create_program('proj', 'prog', code, 'A program', labels) == ('prog', result)
    grpc_client.create_quantum_program.assert_called_with(
        quantum.CreateQuantumProgramRequest(
            parent='projects/proj',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog',
                code=code,
                description='A program',
                labels=labels,
            ),
        )
    )

    assert client.create_program('proj', 'prog', code, 'A program') == ('prog', result)
    grpc_client.create_quantum_program.assert_called_with(
        quantum.CreateQuantumProgramRequest(
            parent='projects/proj',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog', code=code, description='A program'
            ),
        )
    )

    assert client.create_program('proj', 'prog', code, labels=labels) == ('prog', result)
    grpc_client.create_quantum_program.assert_called_with(
        quantum.CreateQuantumProgramRequest(
            parent='projects/proj',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog', code=code, labels=labels
            ),
        )
    )

    assert client.create_program('proj', 'prog', code) == ('prog', result)
    grpc_client.create_quantum_program.assert_called_with(
        quantum.CreateQuantumProgramRequest(
            parent='projects/proj',
            quantum_program=quantum.QuantumProgram(name='projects/proj/programs/prog', code=code),
        )
    )

    assert client.create_program('proj', program_id=None, code=code) == ('prog', result)
    grpc_client.create_quantum_program.assert_called_with(
        quantum.CreateQuantumProgramRequest(
            parent='projects/proj', quantum_program=quantum.QuantumProgram(code=code)
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_program(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.get_quantum_program.return_value = result

    client = EngineClient()
    assert client.get_program('proj', 'prog', False) == result
    grpc_client.get_quantum_program.assert_called_with(
        quantum.GetQuantumProgramRequest(name='projects/proj/programs/prog', return_code=False)
    )

    assert client.get_program('proj', 'prog', True) == result
    grpc_client.get_quantum_program.assert_called_with(
        quantum.GetQuantumProgramRequest(name='projects/proj/programs/prog', return_code=True)
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_program(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    results = [
        quantum.QuantumProgram(name='projects/proj/programs/prog1'),
        quantum.QuantumProgram(name='projects/proj/programs/prog2'),
    ]
    grpc_client.list_quantum_programs.return_value = results

    client = EngineClient()
    assert client.list_programs(project_id='proj') == results
    grpc_client.list_quantum_programs.assert_called_with(
        quantum.ListQuantumProgramsRequest(parent='projects/proj', filter='')
    )


@pytest.mark.parametrize(
    'expected_filter, created_after, created_before, labels',
    [
        ('', None, None, None),
        ('create_time >= 2020-09-01', datetime.date(2020, 9, 1), None, None),
        (
            'create_time >= 1598918400',
            datetime.datetime(2020, 9, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
            None,
            None,
        ),
        ('create_time <= 2020-10-01', None, datetime.date(2020, 10, 1), None),
        (
            'create_time >= 2020-09-01 AND create_time <= 1598918410',
            datetime.date(2020, 9, 1),
            datetime.datetime(2020, 9, 1, 0, 0, 10, tzinfo=datetime.timezone.utc),
            None,
        ),
        ('labels.color:red AND labels.shape:*', None, None, {'color': 'red', 'shape': '*'}),
        (
            'create_time >= 2020-08-01 AND '
            'create_time <= 1598918400 AND '
            'labels.color:red AND labels.shape:*',
            datetime.date(2020, 8, 1),
            datetime.datetime(2020, 9, 1, tzinfo=datetime.timezone.utc),
            {'color': 'red', 'shape': '*'},
        ),
    ],
)
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_program_filters(
    client_constructor, expected_filter, created_before, created_after, labels
):
    grpc_client = _setup_client_mock(client_constructor)
    client = EngineClient()
    client.list_programs(
        project_id='proj',
        created_before=created_before,
        created_after=created_after,
        has_labels=labels,
    )
    assert grpc_client.list_quantum_programs.call_args[0][0].filter == expected_filter


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_program_filters_invalid_type(client_constructor):
    with pytest.raises(ValueError, match=""):
        EngineClient().list_programs(project_id='proj', created_before="Unsupported date/time")


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_set_program_description(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    assert client.set_program_description('proj', 'prog', 'A program') == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog', description='A program'
            ),
            update_mask=FieldMask(paths=['description']),
        )
    )

    assert client.set_program_description('proj', 'prog', '') == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(name='projects/proj/programs/prog'),
            update_mask=FieldMask(paths=['description']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_set_program_labels(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    grpc_client.get_quantum_program.return_value = quantum.QuantumProgram(
        labels={'color': 'red', 'weather': 'sun', 'run': '1'}, label_fingerprint='hash'
    )
    result = quantum.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    labels = {'hello': 'world', 'color': 'blue', 'run': '1'}
    assert client.set_program_labels('proj', 'prog', labels) == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog', labels=labels, label_fingerprint='hash'
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )

    assert client.set_program_labels('proj', 'prog', {}) == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog', label_fingerprint='hash'
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_add_program_labels(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    existing = quantum.QuantumProgram(
        labels={'color': 'red', 'weather': 'sun', 'run': '1'}, label_fingerprint='hash'
    )
    grpc_client.get_quantum_program.return_value = existing
    result = quantum.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    assert client.add_program_labels('proj', 'prog', {'color': 'red'}) == existing
    assert grpc_client.update_quantum_program.call_count == 0

    assert client.add_program_labels('proj', 'prog', {'hello': 'world'}) == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog',
                labels={'color': 'red', 'weather': 'sun', 'run': '1', 'hello': 'world'},
                label_fingerprint='hash',
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )

    assert client.add_program_labels('proj', 'prog', {'hello': 'world', 'color': 'blue'}) == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog',
                labels={'color': 'blue', 'weather': 'sun', 'run': '1', 'hello': 'world'},
                label_fingerprint='hash',
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_remove_program_labels(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    existing = quantum.QuantumProgram(
        labels={'color': 'red', 'weather': 'sun', 'run': '1'}, label_fingerprint='hash'
    )
    grpc_client.get_quantum_program.return_value = existing
    result = quantum.QuantumProgram(name='projects/proj/programs/prog')
    grpc_client.update_quantum_program.return_value = result

    client = EngineClient()
    assert client.remove_program_labels('proj', 'prog', ['other']) == existing
    assert grpc_client.update_quantum_program.call_count == 0

    assert client.remove_program_labels('proj', 'prog', ['hello', 'weather']) == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog',
                labels={'color': 'red', 'run': '1'},
                label_fingerprint='hash',
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )

    assert client.remove_program_labels('proj', 'prog', ['color', 'weather', 'run']) == result
    grpc_client.update_quantum_program.assert_called_with(
        quantum.UpdateQuantumProgramRequest(
            name='projects/proj/programs/prog',
            quantum_program=quantum.QuantumProgram(
                name='projects/proj/programs/prog', label_fingerprint='hash'
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_delete_program(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    client = EngineClient()
    assert not client.delete_program('proj', 'prog')
    grpc_client.delete_quantum_program.assert_called_with(
        quantum.DeleteQuantumProgramRequest(name='projects/proj/programs/prog', delete_jobs=False)
    )

    assert not client.delete_program('proj', 'prog', delete_jobs=True)
    grpc_client.delete_quantum_program.assert_called_with(
        quantum.DeleteQuantumProgramRequest(name='projects/proj/programs/prog', delete_jobs=True)
    )


@mock.patch.dict(os.environ, clear='CIRQ_TESTING')
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_create_job(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.create_quantum_job.return_value = result

    run_context = any_pb2.Any()
    labels = {'hello': 'world'}
    client = EngineClient()
    assert client.create_job(
        'proj', 'prog', 'job0', 'processor0', run_context, 10, 'A job', labels
    ) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(
        quantum.CreateQuantumJobRequest(
            parent='projects/proj/programs/prog',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                run_context=run_context,
                scheduling_config=quantum.SchedulingConfig(
                    priority=10,
                    processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                        processor='projects/proj/processors/processor0',
                        device_config_selector=quantum.DeviceConfigSelector(),
                    ),
                ),
                description='A job',
                labels=labels,
            ),
        )
    )

    assert client.create_job('proj', 'prog', 'job0', 'processor0', run_context, 10, 'A job') == (
        'job0',
        result,
    )
    grpc_client.create_quantum_job.assert_called_with(
        quantum.CreateQuantumJobRequest(
            parent='projects/proj/programs/prog',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                run_context=run_context,
                scheduling_config=quantum.SchedulingConfig(
                    priority=10,
                    processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                        processor='projects/proj/processors/processor0',
                        device_config_selector=quantum.DeviceConfigSelector(),
                    ),
                ),
                description='A job',
            ),
        )
    )

    assert client.create_job(
        'proj', 'prog', 'job0', 'processor0', run_context, 10, labels=labels
    ) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(
        quantum.CreateQuantumJobRequest(
            parent='projects/proj/programs/prog',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                run_context=run_context,
                scheduling_config=quantum.SchedulingConfig(
                    priority=10,
                    processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                        processor='projects/proj/processors/processor0',
                        device_config_selector=quantum.DeviceConfigSelector(),
                    ),
                ),
                labels=labels,
            ),
        )
    )

    assert client.create_job('proj', 'prog', 'job0', 'processor0', run_context, 10) == (
        'job0',
        result,
    )
    grpc_client.create_quantum_job.assert_called_with(
        quantum.CreateQuantumJobRequest(
            parent='projects/proj/programs/prog',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                run_context=run_context,
                scheduling_config=quantum.SchedulingConfig(
                    priority=10,
                    processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                        processor='projects/proj/processors/processor0',
                        device_config_selector=quantum.DeviceConfigSelector(),
                    ),
                ),
            ),
        )
    )

    assert client.create_job(
        'proj', 'prog', job_id=None, processor_id='processor0', run_context=run_context, priority=10
    ) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(
        quantum.CreateQuantumJobRequest(
            parent='projects/proj/programs/prog',
            quantum_job=quantum.QuantumJob(
                run_context=run_context,
                scheduling_config=quantum.SchedulingConfig(
                    priority=10,
                    processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                        processor='projects/proj/processors/processor0',
                        device_config_selector=quantum.DeviceConfigSelector(),
                    ),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match='priority must be between 0 and 1000'):
        client.create_job(
            'proj',
            'prog',
            job_id=None,
            processor_id='processor0',
            run_context=run_context,
            priority=5000,
        )


@mock.patch.dict(os.environ, clear='CIRQ_TESTING')
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
@pytest.mark.parametrize(
    'processor_id, run_name, snapshot_id, device_config_name, error_message',
    [
        ('', '', '', '', 'Must specify a processor id when creating a job.'),
        ('processor0', 'RUN_NAME', '', '', 'Cannot specify only one of top level identifier'),
        ('processor0', '', '', 'CONFIG_ALIAS', 'Cannot specify only one of top level identifier'),
        (
            'processor0',
            'run_name',
            'snapshot_id',
            'CONFIG_ALIAS',
            'Cannot specify both `run_name` and `snapshot_id`',
        ),
    ],
)
def test_create_job_with_invalid_processor_and_device_config_arguments_throws(
    client_constructor, processor_id, run_name, snapshot_id, device_config_name, error_message
):
    grpc_client = _setup_client_mock(client_constructor)
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.create_quantum_job.return_value = result
    client = EngineClient()

    with pytest.raises(ValueError, match=error_message):
        client.create_job(
            project_id='proj',
            program_id='prog',
            job_id=None,
            processor_id=processor_id,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )


@mock.patch.dict(os.environ, clear='CIRQ_TESTING')
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
@pytest.mark.parametrize(
    'run_name, snapshot_id, device_config_name',
    [('RUN_NAME', '', 'CONFIG_NAME'), ('', '', ''), ('', '', '')],
)
def test_create_job_with_run_name_and_device_config_name_succeeds(
    client_constructor, run_name, snapshot_id, device_config_name
):
    grpc_client = _setup_client_mock(client_constructor)
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.create_quantum_job.return_value = result
    run_context = any_pb2.Any()
    client = EngineClient()

    assert client.create_job(
        project_id='proj',
        program_id='prog',
        job_id='job0',
        processor_id="processor0",
        run_name=run_name,
        snapshot_id=snapshot_id,
        device_config_name=device_config_name,
        run_context=run_context,
        priority=10,
    ) == ('job0', result)
    grpc_client.create_quantum_job.assert_called_with(
        quantum.CreateQuantumJobRequest(
            parent='projects/proj/programs/prog',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                run_context=run_context,
                scheduling_config=quantum.SchedulingConfig(
                    priority=10,
                    processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                        processor='projects/proj/processors/processor0',
                        device_config_selector=quantum.DeviceConfigSelector(
                            run_name=run_name or None, config_alias=device_config_name
                        ),
                    ),
                ),
            ),
        )
    )


@mock.patch.dict(os.environ, clear='CIRQ_TESTING')
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_create_job_with_snapshot_id_and_config_successfully_passes_device_config_selector(
    client_constructor,
):
    grpc_client = _setup_client_mock(client_constructor)
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.create_quantum_job.return_value = result
    run_context = any_pb2.Any()
    processor_id = "processor0"
    snapshot_id = "SNAPSHOT_ID"
    device_config_name = "DEVICE_CONFIG_NAME"
    client = EngineClient()

    client.create_job(
        project_id='proj',
        program_id='prog',
        job_id='job0',
        processor_id=processor_id,
        snapshot_id=snapshot_id,
        device_config_name=device_config_name,
        run_context=run_context,
        priority=10,
    )

    job = grpc_client.create_quantum_job.call_args[0][0]
    device_config_selector = (
        job.quantum_job.scheduling_config.processor_selector.device_config_selector
    )
    assert device_config_selector.snapshot_id == snapshot_id
    assert device_config_selector.config_alias == device_config_name


@pytest.mark.parametrize(
    'run_job_kwargs, expected_submit_args',
    [
        (
            {
                'project_id': 'proj',
                'program_id': 'prog',
                'code': any_pb2.Any(),
                'job_id': 'job0',
                'processor_id': 'processor0',
                'run_context': any_pb2.Any(),
                'program_description': 'A program',
                'program_labels': {'hello': 'world'},
                'priority': 10,
                'job_description': 'A job',
                'job_labels': {'hello': 'world'},
            },
            [
                'projects/proj',
                quantum.QuantumProgram(
                    name='projects/proj/programs/prog',
                    code=any_pb2.Any(),
                    description='A program',
                    labels={'hello': 'world'},
                ),
                quantum.QuantumJob(
                    name='projects/proj/programs/prog/jobs/job0',
                    run_context=any_pb2.Any(),
                    scheduling_config=quantum.SchedulingConfig(
                        priority=10,
                        processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                            processor='projects/proj/processors/processor0',
                            device_config_selector=quantum.DeviceConfigSelector(),
                        ),
                    ),
                    description='A job',
                    labels={'hello': 'world'},
                ),
            ],
        ),
        # Missing program labels
        (
            {
                'project_id': 'proj',
                'program_id': 'prog',
                'code': any_pb2.Any(),
                'job_id': 'job0',
                'processor_id': 'processor0',
                'run_context': any_pb2.Any(),
                'program_description': 'A program',
                'priority': 10,
                'job_description': 'A job',
                'job_labels': {'hello': 'world'},
            },
            [
                'projects/proj',
                quantum.QuantumProgram(
                    name='projects/proj/programs/prog', code=any_pb2.Any(), description='A program'
                ),
                quantum.QuantumJob(
                    name='projects/proj/programs/prog/jobs/job0',
                    run_context=any_pb2.Any(),
                    scheduling_config=quantum.SchedulingConfig(
                        priority=10,
                        processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                            processor='projects/proj/processors/processor0',
                            device_config_selector=quantum.DeviceConfigSelector(),
                        ),
                    ),
                    description='A job',
                    labels={'hello': 'world'},
                ),
            ],
        ),
        # Missing program description and labels
        (
            {
                'project_id': 'proj',
                'program_id': 'prog',
                'code': any_pb2.Any(),
                'job_id': 'job0',
                'processor_id': 'processor0',
                'run_context': any_pb2.Any(),
                'priority': 10,
                'job_description': 'A job',
                'job_labels': {'hello': 'world'},
            },
            [
                'projects/proj',
                quantum.QuantumProgram(name='projects/proj/programs/prog', code=any_pb2.Any()),
                quantum.QuantumJob(
                    name='projects/proj/programs/prog/jobs/job0',
                    run_context=any_pb2.Any(),
                    scheduling_config=quantum.SchedulingConfig(
                        priority=10,
                        processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                            processor='projects/proj/processors/processor0',
                            device_config_selector=quantum.DeviceConfigSelector(),
                        ),
                    ),
                    description='A job',
                    labels={'hello': 'world'},
                ),
            ],
        ),
        # Missing job labels
        (
            {
                'project_id': 'proj',
                'program_id': 'prog',
                'code': any_pb2.Any(),
                'job_id': 'job0',
                'processor_id': 'processor0',
                'run_context': any_pb2.Any(),
                'program_description': 'A program',
                'program_labels': {'hello': 'world'},
                'priority': 10,
                'job_description': 'A job',
            },
            [
                'projects/proj',
                quantum.QuantumProgram(
                    name='projects/proj/programs/prog',
                    code=any_pb2.Any(),
                    description='A program',
                    labels={'hello': 'world'},
                ),
                quantum.QuantumJob(
                    name='projects/proj/programs/prog/jobs/job0',
                    run_context=any_pb2.Any(),
                    scheduling_config=quantum.SchedulingConfig(
                        priority=10,
                        processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                            processor='projects/proj/processors/processor0',
                            device_config_selector=quantum.DeviceConfigSelector(),
                        ),
                    ),
                    description='A job',
                ),
            ],
        ),
        # Missing job description and labels
        (
            {
                'project_id': 'proj',
                'program_id': 'prog',
                'code': any_pb2.Any(),
                'job_id': 'job0',
                'processor_id': 'processor0',
                'run_context': any_pb2.Any(),
                'program_description': 'A program',
                'program_labels': {'hello': 'world'},
                'priority': 10,
            },
            [
                'projects/proj',
                quantum.QuantumProgram(
                    name='projects/proj/programs/prog',
                    code=any_pb2.Any(),
                    description='A program',
                    labels={'hello': 'world'},
                ),
                quantum.QuantumJob(
                    name='projects/proj/programs/prog/jobs/job0',
                    run_context=any_pb2.Any(),
                    scheduling_config=quantum.SchedulingConfig(
                        priority=10,
                        processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                            processor='projects/proj/processors/processor0',
                            device_config_selector=quantum.DeviceConfigSelector(),
                        ),
                    ),
                ),
            ],
        ),
        # Missing job priority, description, and labels
        (
            {
                'project_id': 'proj',
                'program_id': 'prog',
                'code': any_pb2.Any(),
                'job_id': 'job0',
                'processor_id': 'processor0',
                'run_context': any_pb2.Any(),
                'program_description': 'A program',
                'program_labels': {'hello': 'world'},
            },
            [
                'projects/proj',
                quantum.QuantumProgram(
                    name='projects/proj/programs/prog',
                    code=any_pb2.Any(),
                    description='A program',
                    labels={'hello': 'world'},
                ),
                quantum.QuantumJob(
                    name='projects/proj/programs/prog/jobs/job0',
                    run_context=any_pb2.Any(),
                    scheduling_config=quantum.SchedulingConfig(
                        processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                            processor='projects/proj/processors/processor0',
                            device_config_selector=quantum.DeviceConfigSelector(),
                        )
                    ),
                ),
            ],
        ),
    ],
)
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
@mock.patch.object(engine_stream_manager, 'StreamManager', autospec=True)
def test_run_job_over_stream(
    manager_constructor, client_constructor, run_job_kwargs, expected_submit_args
):
    _setup_client_mock(client_constructor)
    stream_manager = _setup_stream_manager_mock(manager_constructor)

    parent = expected_submit_args[2].name
    expected_future = duet.futuretools.completed_future(quantum.QuantumResult(parent=parent))
    stream_manager.submit.return_value = expected_future
    client = EngineClient()

    actual_future = client.run_job_over_stream(**run_job_kwargs)

    assert actual_future == expected_future
    stream_manager.submit.assert_called_with(*expected_submit_args)


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
@mock.patch.object(engine_stream_manager, 'StreamManager', autospec=True)
def test_run_job_over_stream_with_snapshot_id_returns_correct_future(
    manager_constructor, client_constructor
):
    _setup_client_mock(client_constructor)
    stream_manager = _setup_stream_manager_mock(manager_constructor)
    client = EngineClient()
    run_job_kwargs = (
        {
            'project_id': 'proj',
            'program_id': 'prog',
            'code': any_pb2.Any(),
            'job_id': 'job0',
            'processor_id': 'processor0',
            'run_context': any_pb2.Any(),
            'snapshot_id': 'SNAPSHOT_ID',
            'device_config_name': 'CONFIG_NAME',
        },
    )

    expected_future = duet.futuretools.completed_future(
        quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    )
    stream_manager.submit.return_value = expected_future
    stream_manager.submit.return_value = expected_future

    actual_future = client.run_job_over_stream(**run_job_kwargs[0])

    assert actual_future == expected_future


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
@mock.patch.object(engine_stream_manager, 'StreamManager', autospec=True)
def test_run_job_over_stream_with_snapshot_id_propogates_snapshot_id(
    manager_constructor, client_constructor
):
    _setup_client_mock(client_constructor)
    stream_manager = _setup_stream_manager_mock(manager_constructor)
    client = EngineClient()
    run_job_kwargs = (
        {
            'project_id': 'proj',
            'program_id': 'prog',
            'code': any_pb2.Any(),
            'job_id': 'job0',
            'processor_id': 'processor0',
            'run_context': any_pb2.Any(),
            'snapshot_id': 'SNAPSHOT_ID',
            'device_config_name': 'CONFIG_NAME',
        },
    )
    expected_submit_args = (
        [
            'projects/proj',
            quantum.QuantumProgram(name='projects/proj/programs/prog', code=any_pb2.Any()),
            quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                run_context=any_pb2.Any(),
                scheduling_config=quantum.SchedulingConfig(
                    processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                        processor='projects/proj/processors/processor0',
                        device_config_selector=quantum.DeviceConfigSelector(
                            snapshot_id="SNAPSHOT_ID", config_alias="CONFIG_NAME"
                        ),
                    )
                ),
            ),
        ],
    )
    parent = expected_submit_args[0][2].name
    expected_future = duet.futuretools.completed_future(quantum.QuantumResult(parent=parent))
    stream_manager.submit.return_value = expected_future
    stream_manager.submit.return_value = expected_future

    _ = client.run_job_over_stream(**run_job_kwargs[0])

    stream_manager.submit.assert_called_with(*expected_submit_args[0])


def test_run_job_over_stream_with_priority_out_of_bound_raises():
    client = EngineClient()

    with pytest.raises(ValueError):
        client.run_job_over_stream(
            project_id='proj',
            program_id='prog',
            code=any_pb2.Any(),
            job_id='job0',
            processor_id='processor0',
            run_context=any_pb2.Any(),
            priority=9001,
        )


def test_run_job_over_stream_processor_unset_raises():
    client = EngineClient()

    with pytest.raises(ValueError, match='Must specify a processor id'):
        client.run_job_over_stream(
            project_id='proj',
            program_id='prog',
            code=any_pb2.Any(),
            job_id='job0',
            processor_id='',
            run_context=any_pb2.Any(),
        )


@pytest.mark.parametrize(
    'run_name, snapshot_id, device_config_name, error_message',
    [
        ('run1', '', '', 'Cannot specify only one of top level identifier'),
        ('', '', 'device_config1', 'Cannot specify only one of top level identifier'),
        ('run', 'snapshot_id', 'config', 'Cannot specify both `run_name` and `snapshot_id`'),
    ],
)
def test_run_job_over_stream_invalid_device_config_raises(
    run_name, snapshot_id, device_config_name, error_message
):
    client = EngineClient()

    with pytest.raises(ValueError, match=error_message):
        client.run_job_over_stream(
            project_id='proj',
            program_id='prog',
            code=any_pb2.Any(),
            job_id='job0',
            processor_id='mysim',
            run_context=any_pb2.Any(),
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_job(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.get_quantum_job.return_value = result

    client = EngineClient()
    assert client.get_job('proj', 'prog', 'job0', False) == result
    grpc_client.get_quantum_job.assert_called_with(
        quantum.GetQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0', return_run_context=False
        )
    )

    assert client.get_job('proj', 'prog', 'job0', True) == result
    grpc_client.get_quantum_job.assert_called_with(
        quantum.GetQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0', return_run_context=True
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_set_job_description(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    assert client.set_job_description('proj', 'prog', 'job0', 'A job') == result
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0', description='A job'
            ),
            update_mask=FieldMask(paths=['description']),
        )
    )

    assert client.set_job_description('proj', 'prog', 'job0', '') == result
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0'),
            update_mask=FieldMask(paths=['description']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_set_job_labels(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    grpc_client.get_quantum_job.return_value = quantum.QuantumJob(
        labels={'color': 'red', 'weather': 'sun', 'run': '1'}, label_fingerprint='hash'
    )
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    labels = {'hello': 'world', 'color': 'blue', 'run': '1'}
    assert client.set_job_labels('proj', 'prog', 'job0', labels) == result
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                labels=labels,
                label_fingerprint='hash',
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )

    assert client.set_job_labels('proj', 'prog', 'job0', {}) == result
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0', label_fingerprint='hash'
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_add_job_labels(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    existing = quantum.QuantumJob(
        labels={'color': 'red', 'weather': 'sun', 'run': '1'}, label_fingerprint='hash'
    )
    grpc_client.get_quantum_job.return_value = existing
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    assert client.add_job_labels('proj', 'prog', 'job0', {'color': 'red'}) == existing
    assert grpc_client.update_quantum_job.call_count == 0

    assert client.add_job_labels('proj', 'prog', 'job0', {'hello': 'world'}) == result
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                labels={'color': 'red', 'weather': 'sun', 'run': '1', 'hello': 'world'},
                label_fingerprint='hash',
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )

    assert (
        client.add_job_labels('proj', 'prog', 'job0', {'hello': 'world', 'color': 'blue'}) == result
    )
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                labels={'color': 'blue', 'weather': 'sun', 'run': '1', 'hello': 'world'},
                label_fingerprint='hash',
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_remove_job_labels(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    existing = quantum.QuantumJob(
        labels={'color': 'red', 'weather': 'sun', 'run': '1'}, label_fingerprint='hash'
    )
    grpc_client.get_quantum_job.return_value = existing
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result

    client = EngineClient()
    assert client.remove_job_labels('proj', 'prog', 'job0', ['other']) == existing
    assert grpc_client.update_quantum_program.call_count == 0

    assert client.remove_job_labels('proj', 'prog', 'job0', ['hello', 'weather']) == result
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0',
                labels={'color': 'red', 'run': '1'},
                label_fingerprint='hash',
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )

    assert client.remove_job_labels('proj', 'prog', 'job0', ['color', 'weather', 'run']) == result
    grpc_client.update_quantum_job.assert_called_with(
        quantum.UpdateQuantumJobRequest(
            name='projects/proj/programs/prog/jobs/job0',
            quantum_job=quantum.QuantumJob(
                name='projects/proj/programs/prog/jobs/job0', label_fingerprint='hash'
            ),
            update_mask=FieldMask(paths=['labels']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_delete_job(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    client = EngineClient()
    assert not client.delete_job('proj', 'prog', 'job0')
    grpc_client.delete_quantum_job.assert_called_with(
        quantum.DeleteQuantumJobRequest(name='projects/proj/programs/prog/jobs/job0')
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_cancel_job(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    client = EngineClient()
    assert not client.cancel_job('proj', 'prog', 'job0')
    grpc_client.cancel_quantum_job.assert_called_with(
        quantum.CancelQuantumJobRequest(name='projects/proj/programs/prog/jobs/job0')
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_job_results(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    grpc_client.get_quantum_result.return_value = result

    client = EngineClient()
    assert client.get_job_results('proj', 'prog', 'job0') == result
    grpc_client.get_quantum_result.assert_called_with(
        quantum.GetQuantumResultRequest(parent='projects/proj/programs/prog/jobs/job0')
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_jobs(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    results = [
        quantum.QuantumJob(name='projects/proj/programs/prog1/jobs/job1'),
        quantum.QuantumJob(name='projects/proj/programs/prog1/jobs/job2'),
    ]
    grpc_client.list_quantum_jobs.return_value = results

    client = EngineClient()
    assert client.list_jobs(project_id='proj', program_id='prog1') == results
    grpc_client.list_quantum_jobs.assert_called_with(
        quantum.ListQuantumJobsRequest(parent='projects/proj/programs/prog1')
    )

    assert client.list_jobs(project_id='proj') == results
    grpc_client.list_quantum_jobs.assert_called_with(
        quantum.ListQuantumJobsRequest(parent='projects/proj/programs/-')
    )


@pytest.mark.parametrize(
    'expected_filter, '
    'created_after, '
    'created_before, '
    'labels, '
    'execution_states, '
    'executed_processor_ids, '
    'scheduled_processor_ids, ',
    [
        ('', None, None, None, None, None, None),
        ('create_time >= 2020-09-01', datetime.date(2020, 9, 1), None, None, None, None, None),
        (
            'create_time >= 1598918400',
            datetime.datetime(2020, 9, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
            None,
            None,
            None,
            None,
            None,
        ),
        ('create_time <= 2020-10-01', None, datetime.date(2020, 10, 1), None, None, None, None),
        (
            'create_time >= 2020-09-01 AND create_time <= 1598918410',
            datetime.date(2020, 9, 1),
            datetime.datetime(2020, 9, 1, 0, 0, 10, tzinfo=datetime.timezone.utc),
            None,
            None,
            None,
            None,
        ),
        (
            'labels.color:red AND labels.shape:*',
            None,
            None,
            {'color': 'red', 'shape': '*'},
            None,
            None,
            None,
        ),
        (
            '(execution_status.state = FAILURE OR ' 'execution_status.state = CANCELLED)',
            None,
            None,
            None,
            [quantum.ExecutionStatus.State.FAILURE, quantum.ExecutionStatus.State.CANCELLED],
            None,
            None,
        ),
        (
            'create_time >= 2020-08-01 AND '
            'create_time <= 1598918400 AND '
            'labels.color:red AND labels.shape:* AND '
            '(execution_status.state = SUCCESS)',
            datetime.date(2020, 8, 1),
            datetime.datetime(2020, 9, 1, tzinfo=datetime.timezone.utc),
            {'color': 'red', 'shape': '*'},
            [quantum.ExecutionStatus.State.SUCCESS],
            None,
            None,
        ),
        ('(executed_processor_id = proc1)', None, None, None, None, ['proc1'], None),
        (
            '(executed_processor_id = proc1 OR executed_processor_id = proc2)',
            None,
            None,
            None,
            None,
            ['proc1', 'proc2'],
            None,
        ),
        ('(scheduled_processor_ids: proc1)', None, None, None, None, None, ['proc1']),
        (
            '(scheduled_processor_ids: proc1 OR scheduled_processor_ids: proc2)',
            None,
            None,
            None,
            None,
            None,
            ['proc1', 'proc2'],
        ),
    ],
)
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_jobs_filters(
    client_constructor,
    expected_filter,
    created_before,
    created_after,
    labels,
    execution_states,
    executed_processor_ids,
    scheduled_processor_ids,
):
    grpc_client = _setup_client_mock(client_constructor)
    client = EngineClient()
    client.list_jobs(
        project_id='proj',
        program_id='prog',
        created_before=created_before,
        created_after=created_after,
        has_labels=labels,
        execution_states=execution_states,
        executed_processor_ids=executed_processor_ids,
        scheduled_processor_ids=scheduled_processor_ids,
    )
    assert grpc_client.list_quantum_jobs.call_args[0][0].filter == expected_filter


class Pager:
    """An asyncio AsyncIterator over a fixed list of results."""

    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            await asyncio.sleep(0.001)
            yield item


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_processors(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    results = [
        quantum.QuantumProcessor(name='projects/proj/processor/processor0'),
        quantum.QuantumProcessor(name='projects/proj/processor/processor1'),
    ]
    grpc_client.list_quantum_processors.return_value = Pager(results)

    client = EngineClient()
    assert client.list_processors('proj') == results
    grpc_client.list_quantum_processors.assert_called_with(
        quantum.ListQuantumProcessorsRequest(parent='projects/proj')
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_processor(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumProcessor(name='projects/proj/processors/processor0')
    grpc_client.get_quantum_processor.return_value = result

    client = EngineClient()
    assert client.get_processor('proj', 'processor0') == result
    grpc_client.get_quantum_processor.assert_called_with(
        quantum.GetQuantumProcessorRequest(name='projects/proj/processors/processor0')
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_calibrations(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    results = [
        quantum.QuantumCalibration(name='projects/proj/processor/processor0/calibrations/123456'),
        quantum.QuantumCalibration(name='projects/proj/processor/processor1/calibrations/224466'),
    ]
    grpc_client.list_quantum_calibrations.return_value = Pager(results)

    client = EngineClient()
    assert client.list_calibrations('proj', 'processor0') == results
    grpc_client.list_quantum_calibrations.assert_called_with(
        quantum.ListQuantumCalibrationsRequest(parent='projects/proj/processors/processor0')
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_calibration(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumCalibration(
        name='projects/proj/processors/processor0/calibrations/123456'
    )
    grpc_client.get_quantum_calibration.return_value = result

    client = EngineClient()
    assert client.get_calibration('proj', 'processor0', 123456) == result
    grpc_client.get_quantum_calibration.assert_called_with(
        quantum.GetQuantumCalibrationRequest(
            name='projects/proj/processors/processor0/calibrations/123456'
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_current_calibration(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    result = quantum.QuantumCalibration(
        name='projects/proj/processors/processor0/calibrations/123456'
    )
    grpc_client.get_quantum_calibration.return_value = result

    client = EngineClient()
    assert client.get_current_calibration('proj', 'processor0') == result
    grpc_client.get_quantum_calibration.assert_called_with(
        quantum.GetQuantumCalibrationRequest(
            name='projects/proj/processors/processor0/calibrations/current'
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_current_calibration_does_not_exist(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    grpc_client.get_quantum_calibration.side_effect = exceptions.NotFound('not found')

    client = EngineClient()
    assert client.get_current_calibration('proj', 'processor0') is None
    grpc_client.get_quantum_calibration.assert_called_with(
        quantum.GetQuantumCalibrationRequest(
            name='projects/proj/processors/processor0/calibrations/current'
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_current_calibration_error(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)

    grpc_client.get_quantum_calibration.side_effect = exceptions.BadRequest('boom')

    client = EngineClient()
    with pytest.raises(EngineException, match='boom'):
        client.get_current_calibration('proj', 'processor0')


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_api_doesnt_retry_not_found_errors(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.NotFound('not found')

    client = EngineClient()
    with pytest.raises(EngineException, match='not found'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 1


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_api_retry_5xx_errors(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.ServiceUnavailable('internal error')

    client = EngineClient(max_retry_delay_seconds=0.3)
    with pytest.raises(TimeoutError, match='Reached max retry attempts.*internal error'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 3


@mock.patch('duet.sleep', return_value=duet.completed_future(None))
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_api_retry_times(client_constructor, mock_sleep):
    grpc_client = _setup_client_mock(client_constructor)
    grpc_client.get_quantum_program.side_effect = exceptions.ServiceUnavailable('internal error')

    client = EngineClient(max_retry_delay_seconds=0.3)
    with pytest.raises(TimeoutError, match='Reached max retry attempts.*internal error'):
        client.get_program('proj', 'prog', False)
    assert grpc_client.get_quantum_program.call_count == 3

    assert len(mock_sleep.call_args_list) == 2
    assert all(x == y for (x, _), y in zip(mock_sleep.call_args_list, [(0.1,), (0.2,)]))


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_create_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    start = datetime.datetime.fromtimestamp(1000000000)
    end = datetime.datetime.fromtimestamp(1000003600)
    users = ['jeff@google.com']
    result = quantum.QuantumReservation(
        name='projects/proj/processors/processor0/reservations/papar-party-44',
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=users,
    )
    grpc_client.create_quantum_reservation.return_value = result

    client = EngineClient()
    assert client.create_reservation('proj', 'processor0', start, end, users) == result
    assert grpc_client.create_quantum_reservation.call_count == 1
    # The outgoing argument will not have the resource name
    result.name = ''
    grpc_client.create_quantum_reservation.assert_called_with(
        quantum.CreateQuantumReservationRequest(
            parent='projects/proj/processors/processor0', quantum_reservation=result
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_cancel_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.cancel_quantum_reservation.return_value = result

    client = EngineClient()
    assert client.cancel_reservation('proj', 'processor0', 'papar-party-44') == result
    grpc_client.cancel_quantum_reservation.assert_called_with(
        quantum.CancelQuantumReservationRequest(name=name)
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_delete_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.delete_quantum_reservation.return_value = result

    client = EngineClient()
    assert client.delete_reservation('proj', 'processor0', 'papar-party-44') == result
    grpc_client.delete_quantum_reservation.assert_called_with(
        quantum.DeleteQuantumReservationRequest(name=name)
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.get_quantum_reservation.return_value = result

    client = EngineClient()
    assert client.get_reservation('proj', 'processor0', 'papar-party-44') == result
    grpc_client.get_quantum_reservation.assert_called_with(
        quantum.GetQuantumReservationRequest(name=name)
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_reservation_not_found(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    grpc_client.get_quantum_reservation.side_effect = exceptions.NotFound('not found')

    client = EngineClient()
    assert client.get_reservation('proj', 'processor0', 'papar-party-44') is None
    grpc_client.get_quantum_reservation.assert_called_with(
        quantum.GetQuantumReservationRequest(name=name)
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_get_reservation_exception(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    grpc_client.get_quantum_reservation.side_effect = exceptions.BadRequest('boom')

    client = EngineClient()
    with pytest.raises(EngineException, match='boom'):
        client.get_reservation('proj', 'processor0', 'goog')


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    results = [
        quantum.QuantumReservation(
            name=name,
            start_time=Timestamp(seconds=1000000000),
            end_time=Timestamp(seconds=1000002000),
            whitelisted_users=['jeff@google.com'],
        ),
        quantum.QuantumReservation(
            name=name,
            start_time=Timestamp(seconds=1200000000),
            end_time=Timestamp(seconds=1200002000),
            whitelisted_users=['dstrain@google.com'],
        ),
    ]
    grpc_client.list_quantum_reservations.return_value = Pager(results)

    client = EngineClient()
    assert client.list_reservations('proj', 'processor0') == results


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_update_reservation(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000001000),
        end_time=Timestamp(seconds=1000002000),
        whitelisted_users=['jeff@google.com'],
    )
    grpc_client.update_quantum_reservation.return_value = result

    client = EngineClient()
    assert (
        client.update_reservation(
            'proj',
            'processor0',
            'papar-party-44',
            start=datetime.datetime.fromtimestamp(1000001000),
            end=datetime.datetime.fromtimestamp(1000002000),
            whitelisted_users=['jeff@google.com'],
        )
        == result
    )
    grpc_client.update_quantum_reservation.assert_called_with(
        quantum.UpdateQuantumReservationRequest(
            name=name,
            quantum_reservation=result,
            update_mask=FieldMask(paths=['start_time', 'end_time', 'whitelisted_users']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_update_reservation_remove_all_users(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    name = 'projects/proj/processors/processor0/reservations/papar-party-44'
    result = quantum.QuantumReservation(name=name, whitelisted_users=[])
    grpc_client.update_quantum_reservation.return_value = result

    client = EngineClient()
    assert (
        client.update_reservation('proj', 'processor0', 'papar-party-44', whitelisted_users=[])
        == result
    )
    grpc_client.update_quantum_reservation.assert_called_with(
        quantum.UpdateQuantumReservationRequest(
            name=name,
            quantum_reservation=result,
            update_mask=FieldMask(paths=['whitelisted_users']),
        )
    )


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_list_time_slots(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    results = [
        quantum.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000020000),
            end_time=Timestamp(seconds=1000040000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            maintenance_config=quantum.QuantumTimeSlot.MaintenanceConfig(
                title='Testing', description='Testing some new configuration.'
            ),
        ),
        quantum.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000010000),
            end_time=Timestamp(seconds=1000020000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
            reservation_config=quantum.QuantumTimeSlot.ReservationConfig(
                project_id='super_secret_quantum'
            ),
        ),
    ]
    grpc_client.list_quantum_time_slots.return_value = Pager(results)

    client = EngineClient()
    assert client.list_time_slots('proj', 'processor0') == results

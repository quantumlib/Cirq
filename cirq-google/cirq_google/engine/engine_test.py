# Copyright 2018 The Cirq Developers
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

"""Tests for engine."""
import datetime
from unittest import mock
import time
import numpy as np
import pytest

import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge

import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext

_CIRCUIT = cirq.Circuit(
    cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
)


_CIRCUIT2 = cirq.FrozenCircuit(
    cirq.Y(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
)


def _to_timestamp(json_string):
    timestamp_proto = timestamp_pb2.Timestamp()
    timestamp_proto.FromJsonString(json_string)
    return timestamp_proto


_A_RESULT = util.pack_any(
    Merge(
        """
sweep_results: [{
        repetitions: 1,
        measurement_keys: [{
            key: 'q',
            qubits: [{
                row: 1,
                col: 1
            }]
        }],
        parameterized_results: [{
            params: {
                assignments: {
                    key: 'a'
                    value: 1
                }
            },
            measurement_results: '\000\001'
        }]
    }]
""",
        v1.program_pb2.Result(),
    )
)

_RESULTS = util.pack_any(
    Merge(
        """
sweep_results: [{
        repetitions: 1,
        measurement_keys: [{
            key: 'q',
            qubits: [{
                row: 1,
                col: 1
            }]
        }],
        parameterized_results: [{
            params: {
                assignments: {
                    key: 'a'
                    value: 1
                }
            },
            measurement_results: '\000\001'
        },{
            params: {
                assignments: {
                    key: 'a'
                    value: 2
                }
            },
            measurement_results: '\000\001'
        }]
    }]
""",
        v1.program_pb2.Result(),
    )
)

_RESULTS_V2 = util.pack_any(
    Merge(
        """
sweep_results: [{
        repetitions: 1,
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
                  results: '\000\001'
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
                  results: '\000\001'
                }]
            }
        }]
    }]
""",
        v2.result_pb2.Result(),
    )
)


def test_make_random_id():
    with mock.patch('random.choice', return_value='A'):
        random_id = cg.engine.engine._make_random_id('prefix-', length=4)
        assert random_id[:11] == 'prefix-AAAA'
    random_id = cg.engine.engine._make_random_id('prefix-')
    time.sleep(1)
    random_id2 = cg.engine.engine._make_random_id('prefix-')
    # Verify %H%M%S is populated
    assert random_id[-7:] != '-000000' or random_id2[-7:] != '-000000'
    # Verify program id generate distinct even if random is seeded
    assert random_id != random_id2


@pytest.fixture(scope='module', autouse=True)
def mock_grpc_client_async():
    with mock.patch(
        'cirq_google.engine.engine_client.quantum.QuantumEngineServiceAsyncClient', autospec=True
    ) as _fixture:
        yield _fixture


@mock.patch('cirq_google.engine.engine_client.EngineClient')
def test_create_context(client):
    with pytest.raises(ValueError, match='specify service_args and verbose or client'):
        EngineContext(cg.engine.engine.ProtoVersion.V1, {'args': 'test'}, True, mock.Mock())
    with pytest.raises(ValueError, match='no longer supported'):
        _ = EngineContext(cg.engine.engine.ProtoVersion.V1, {'args': 'test'}, True)

    context = EngineContext(cg.engine.engine.ProtoVersion.V2, {'args': 'test'}, True)
    assert context.proto_version == cg.engine.engine.ProtoVersion.V2
    client.assert_called_with(service_args={'args': 'test'}, verbose=True)

    assert context.copy().proto_version == context.proto_version
    assert context.copy().client == context.client
    assert context.copy() == context


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_create_engine(client):
    with pytest.raises(
        ValueError, match='provide context or proto_version, service_args and verbose'
    ):
        cg.Engine(
            'proj',
            proto_version=cg.engine.engine.ProtoVersion.V2,
            service_args={'args': 'test'},
            verbose=True,
            context=mock.Mock(),
        )

    assert (
        cg.Engine(
            'proj',
            proto_version=cg.engine.engine.ProtoVersion.V2,
            service_args={'args': 'test'},
            verbose=True,
        ).context.proto_version
        == cg.engine.engine.ProtoVersion.V2
    )
    client.assert_called_with({'args': 'test'}, True)


def test_engine_str():
    engine = cg.Engine(
        'proj', proto_version=cg.engine.engine.ProtoVersion.V2, service_args={}, verbose=True
    )
    assert str(engine) == "Engine(project_id='proj')"


_DT = datetime.datetime.now(tz=datetime.timezone.utc)


def setup_run_circuit_with_result_(client, result):
    client().create_program_async.return_value = (
        'prog',
        quantum.QuantumProgram(name='projects/proj/programs/prog'),
    )
    client().create_job_async.return_value = (
        'job-id',
        quantum.QuantumJob(
            name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}
        ),
    )
    client().get_job_async.return_value = quantum.QuantumJob(
        execution_status={'state': 'SUCCESS'}, update_time=_DT
    )
    client().get_job_results_async.return_value = quantum.QuantumResult(result=result)
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(quantum.QuantumResult(result=result))
    client().run_job_over_stream.return_value = stream_future


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _A_RESULT)

    engine = cg.Engine(
        project_id='proj',
        context=EngineContext(service_args={'client_info': 1}, enable_streaming=False),
    )
    result = engine.run(program=_CIRCUIT, program_id='prog', job_id='job-id', processor_id='mysim')

    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    client.assert_called_with(service_args={'client_info': 1}, verbose=None)
    client().create_program_async.assert_called_once()
    client().create_job_async.assert_called_once_with(
        project_id='proj',
        program_id='prog',
        job_id='job-id',
        processor_id='mysim',
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=1)]
            )
        ),
        description=None,
        labels=None,
        run_name='',
        snapshot_id='',
        device_config_name='',
    )
    client().get_job_async.assert_called_once_with('proj', 'prog', 'job-id', False)
    client().get_job_results_async.assert_called_once_with('proj', 'prog', 'job-id')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_engine_get_sampler_with_snapshot_id_passes_to_unary_rpc(client):
    setup_run_circuit_with_result_(client, _A_RESULT)
    engine = cg.Engine(
        project_id='proj',
        context=EngineContext(service_args={'client_info': 1}, enable_streaming=False),
    )
    sampler = engine.get_sampler('mysim', device_config_name="config", snapshot_id="123")
    _ = sampler.run_sweep(_CIRCUIT, params=[cirq.ParamResolver({'a': 1})])

    kwargs = client().create_job_async.call_args_list[0].kwargs

    # We care about asserting that the snapshot_id is correctly passed.
    assert kwargs["snapshot_id"] == "123"
    assert kwargs["run_name"] == ""
    assert kwargs["device_config_name"] == "config"


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_with_stream_rpcs_passes(client):
    setup_run_circuit_with_result_(client, _A_RESULT)

    engine = cg.Engine(
        project_id='proj',
        context=EngineContext(service_args={'client_info': 1}, enable_streaming=True),
    )
    result = engine.run(program=_CIRCUIT, program_id='prog', job_id='job-id', processor_id='mysim')

    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    client.assert_called_with(service_args={'client_info': 1}, verbose=None)
    client().run_job_over_stream.assert_called_once_with(
        project_id='proj',
        program_id='prog',
        code=mock.ANY,
        job_id='job-id',
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=1)]
            )
        ),
        program_description=None,
        program_labels=None,
        job_description=None,
        job_labels=None,
        processor_id='mysim',
        run_name='',
        snapshot_id='',
        device_config_name='',
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_snapshot_id_with_stream_rpcs(client):
    setup_run_circuit_with_result_(client, _A_RESULT)

    engine = cg.Engine(
        project_id='proj',
        context=EngineContext(service_args={'client_info': 1}, enable_streaming=True),
    )
    result = engine.run(
        program=_CIRCUIT,
        program_id='prog',
        job_id='job-id',
        processor_id='mysim',
        snapshot_id="123",
        device_config_name="config",
    )

    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    client.assert_called_with(service_args={'client_info': 1}, verbose=None)
    client().run_job_over_stream.assert_called_once_with(
        project_id='proj',
        program_id='prog',
        code=mock.ANY,
        job_id='job-id',
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=1)]
            )
        ),
        program_description=None,
        program_labels=None,
        job_description=None,
        job_labels=None,
        processor_id='mysim',
        run_name='',
        snapshot_id="123",
        device_config_name='config',
    )


def test_no_gate_set():
    engine = cg.Engine(project_id='project-id')
    assert engine.context.serializer == cg.CIRCUIT_SERIALIZER


def test_unsupported_program_type():
    engine = cg.Engine(project_id='project-id')
    with pytest.raises(TypeError, match='program'):
        engine.run(program="this isn't even the right type of thing!", processor_id='processor0')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_failed_with_unary_rpcs(client):
    client().create_program_async.return_value = (
        'prog',
        quantum.QuantumProgram(name='projects/proj/programs/prog'),
    )
    client().create_job_async.return_value = (
        'job-id',
        quantum.QuantumJob(
            name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}
        ),
    )
    client().get_job_async.return_value = quantum.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'FAILURE',
            'processor_name': 'myqc',
            'failure': {'error_code': 'SYSTEM_ERROR', 'error_message': 'Not good'},
        },
    )

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    with pytest.raises(
        RuntimeError,
        match='Job projects/proj/programs/prog/jobs/job-id on processor'
        ' myqc failed. SYSTEM_ERROR: Not good',
    ):
        engine.run(program=_CIRCUIT, processor_id='processor0')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_failed_with_stream_rpcs(client):
    failed_job = quantum.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'FAILURE',
            'processor_name': 'myqc',
            'failure': {'error_code': 'SYSTEM_ERROR', 'error_message': 'Not good'},
        },
    )
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(failed_job)
    client().run_job_over_stream.return_value = stream_future

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=True))
    with pytest.raises(
        RuntimeError,
        match='Job projects/proj/programs/prog/jobs/job-id on processor'
        ' myqc failed. SYSTEM_ERROR: Not good',
    ):
        engine.run(program=_CIRCUIT, processor_id='processor0')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_failed_missing_processor_name_with_unary_rpcs(client):
    client().create_program_async.return_value = (
        'prog',
        quantum.QuantumProgram(name='projects/proj/programs/prog'),
    )
    client().create_job_async.return_value = (
        'job-id',
        quantum.QuantumJob(
            name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}
        ),
    )
    client().get_job_async.return_value = quantum.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'FAILURE',
            'failure': {'error_code': 'SYSTEM_ERROR', 'error_message': 'Not good'},
        },
    )

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    with pytest.raises(
        RuntimeError,
        match='Job projects/proj/programs/prog/jobs/job-id on processor'
        ' UNKNOWN failed. SYSTEM_ERROR: Not good',
    ):
        engine.run(program=_CIRCUIT, processor_id='processor0')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_failed_missing_processor_name_with_stream_rpcs(client):
    failed_job = quantum.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'FAILURE',
            'failure': {'error_code': 'SYSTEM_ERROR', 'error_message': 'Not good'},
        },
    )
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(failed_job)
    client().run_job_over_stream.return_value = stream_future

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=True))
    with pytest.raises(
        RuntimeError,
        match='Job projects/proj/programs/prog/jobs/job-id on processor'
        ' UNKNOWN failed. SYSTEM_ERROR: Not good',
    ):
        engine.run(program=_CIRCUIT, processor_id='processor0')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_cancelled_with_unary_rpcs(client):
    client().create_program_async.return_value = (
        'prog',
        quantum.QuantumProgram(name='projects/proj/programs/prog'),
    )
    client().create_job_async.return_value = (
        'job-id',
        quantum.QuantumJob(
            name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}
        ),
    )
    client().get_job_async.return_value = quantum.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'CANCELLED'}
    )

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    with pytest.raises(
        RuntimeError, match='Job projects/proj/programs/prog/jobs/job-id failed in state CANCELLED.'
    ):
        engine.run(program=_CIRCUIT, processor_id='processor0')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_circuit_cancelled_with_stream_rpcs(client):
    canceled_job = quantum.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'CANCELLED'}
    )
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(canceled_job)
    client().run_job_over_stream.return_value = stream_future

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=True))
    with pytest.raises(
        RuntimeError, match='Job projects/proj/programs/prog/jobs/job-id failed in state CANCELLED.'
    ):
        engine.run(program=_CIRCUIT, processor_id='processor0')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_sweep_params_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    job = engine.run_sweep(
        program=_CIRCUIT,
        processor_id='processor0',
        params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})],
    )
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}

    client().create_program_async.assert_called_once()
    client().create_job_async.assert_called_once()

    run_context = v2.run_context_pb2.RunContext()
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 2
    for i, v in enumerate([1, 2]):
        assert sweeps[i].repetitions == 1
        assert sweeps[i].sweep.sweep_function.sweeps[0].single_sweep.const_value.int_value == v
    client().get_job_async.assert_called_once()
    client().get_job_results_async.assert_called_once()


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_sweep_params_with_stream_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=True))
    job = engine.run_sweep(
        program=_CIRCUIT,
        processor_id='processor0',
        params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})],
    )
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}

    client().run_job_over_stream.assert_called_once()

    run_context = v2.run_context_pb2.RunContext()
    client().run_job_over_stream.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 2
    for i, v in enumerate([1, 2]):
        assert sweeps[i].repetitions == 1
        assert sweeps[i].sweep.sweep_function.sweeps[0].single_sweep.const_value.int_value == v


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_multiple_times(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj', proto_version=cg.engine.engine.ProtoVersion.V2)
    program = engine.create_program(program=_CIRCUIT)
    program.run(
        processor_id='processor0',
        param_resolver=cirq.ParamResolver({'a': 1}),
        run_name="run",
        device_config_name="config_alias",
    )
    run_context = v2.run_context_pb2.RunContext()
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps1 = run_context.parameter_sweeps
    job2 = program.run_sweep(
        processor_id='processor0',
        repetitions=2,
        params=cirq.Points('a', [3, 4]),
        run_name="run",
        device_config_name="config_alias",
    )
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps2 = run_context.parameter_sweeps
    results = job2.results()
    assert engine.context.proto_version == cg.engine.engine.ProtoVersion.V2
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert len(sweeps1) == 1
    assert sweeps1[0].repetitions == 1
    assert sweeps1[0].sweep.sweep_function.sweeps[0].single_sweep.const_value.int_value == 1
    assert len(sweeps2) == 1
    assert sweeps2[0].repetitions == 2
    assert sweeps2[0].sweep.single_sweep.points.points == [3, 4]
    assert client().get_job_async.call_count == 2
    assert client().get_job_results_async.call_count == 2


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_sweep_v2_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS_V2)

    engine = cg.Engine(
        project_id='proj',
        context=EngineContext(
            proto_version=cg.engine.engine.ProtoVersion.V2, enable_streaming=False
        ),
    )
    job = engine.run_sweep(
        program=_CIRCUIT,
        processor_id='processor0',
        job_id='job-id',
        params=cirq.Points('a', [1, 2]),
    )
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    client().create_program_async.assert_called_once()
    client().create_job_async.assert_called_once()
    run_context = v2.run_context_pb2.RunContext()
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 1
    assert sweeps[0].repetitions == 1
    assert sweeps[0].sweep.single_sweep.points.points == [1, 2]
    client().get_job_async.assert_called_once()
    client().get_job_results_async.assert_called_once()


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_sweep_v2_with_stream_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS_V2)

    engine = cg.Engine(
        project_id='proj',
        context=EngineContext(
            proto_version=cg.engine.engine.ProtoVersion.V2, enable_streaming=True
        ),
    )
    job = engine.run_sweep(
        program=_CIRCUIT,
        processor_id='processor0',
        job_id='job-id',
        params=cirq.Points('a', [1, 2]),
    )
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    client().run_job_over_stream.assert_called_once()
    run_context = v2.run_context_pb2.RunContext()
    client().run_job_over_stream.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 1
    assert sweeps[0].repetitions == 1
    assert sweeps[0].sweep.single_sweep.points.points == [1, 2]


def test_bad_sweep_proto():
    engine = cg.Engine(project_id='project-id', proto_version=cg.ProtoVersion.UNDEFINED)
    program = cg.EngineProgram('proj', 'prog', engine.context)
    with pytest.raises(ValueError, match='invalid run context proto version'):
        program.run_sweep(
            processor_id='processor0', run_name="run", device_config_name="config_alias"
        )


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_bad_result_proto(client):
    result = any_pb2.Any()
    result.CopyFrom(_RESULTS_V2)
    result.type_url = 'type.googleapis.com/unknown'
    setup_run_circuit_with_result_(client, result)

    engine = cg.Engine(project_id='project-id', proto_version=cg.engine.engine.ProtoVersion.V2)
    job = engine.run_sweep(
        program=_CIRCUIT,
        processor_id='processor0',
        job_id='job-id',
        params=cirq.Points('a', [1, 2]),
    )
    with pytest.raises(ValueError, match='invalid result proto version'):
        job.results()


def test_bad_program_proto():
    engine = cg.Engine(
        project_id='project-id', proto_version=cg.engine.engine.ProtoVersion.UNDEFINED
    )
    with pytest.raises(ValueError, match='invalid (program|run context) proto version'):
        engine.run_sweep(program=_CIRCUIT, processor_id='processor0')
    with pytest.raises(ValueError, match='invalid program proto version'):
        engine.create_program(_CIRCUIT)


def test_get_program():
    assert cg.Engine(project_id='proj').get_program('prog').program_id == 'prog'


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_programs_async')
def test_list_programs(list_programs_async):
    prog1 = quantum.QuantumProgram(name='projects/proj/programs/prog-YBGR48THF3JHERZW200804')
    prog2 = quantum.QuantumProgram(name='projects/otherproj/programs/prog-V3ZRTV6TTAFNTYJV200804')
    list_programs_async.return_value = [prog1, prog2]

    result = cg.Engine(project_id='proj').list_programs()
    list_programs_async.assert_called_once_with(
        'proj', created_after=None, created_before=None, has_labels=None
    )
    assert [(p.program_id, p.project_id, p._program) for p in result] == [
        ('prog-YBGR48THF3JHERZW200804', 'proj', prog1),
        ('prog-V3ZRTV6TTAFNTYJV200804', 'otherproj', prog2),
    ]


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_create_program(client):
    client().create_program_async.return_value = ('prog', quantum.QuantumProgram())
    result = cg.Engine(project_id='proj').create_program(_CIRCUIT, 'prog')
    client().create_program_async.assert_called_once()
    assert result.program_id == 'prog'


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_jobs_async')
def test_list_jobs(list_jobs_async):
    job1 = quantum.QuantumJob(name='projects/proj/programs/prog1/jobs/job1')
    job2 = quantum.QuantumJob(name='projects/proj/programs/prog2/jobs/job2')
    list_jobs_async.return_value = [job1, job2]

    ctx = EngineContext()
    result = cg.Engine(project_id='proj', context=ctx).list_jobs()
    list_jobs_async.assert_called_once_with(
        'proj',
        None,
        created_after=None,
        created_before=None,
        has_labels=None,
        execution_states=None,
    )
    assert [(j.project_id, j.program_id, j.job_id, j.context, j._job) for j in result] == [
        ('proj', 'prog1', 'job1', ctx, job1),
        ('proj', 'prog2', 'job2', ctx, job2),
    ]


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_processors_async')
def test_list_processors(list_processors_async):
    processor1 = quantum.QuantumProcessor(name='projects/proj/processors/xmonsim')
    processor2 = quantum.QuantumProcessor(name='projects/proj/processors/gmonsim')
    list_processors_async.return_value = [processor1, processor2]

    result = cg.Engine(project_id='proj').list_processors()
    list_processors_async.assert_called_once_with('proj')
    assert [p.processor_id for p in result] == ['xmonsim', 'gmonsim']


def test_get_processor():
    assert cg.Engine(project_id='proj').get_processor('xmonsim').processor_id == 'xmonsim'


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_sampler_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    sampler = engine.get_sampler(processor_id='tmp')
    results = sampler.run_sweep(
        program=_CIRCUIT, params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})]
    )
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client().create_program_async.call_args[0][0] == 'proj'

    with pytest.raises(ValueError, match='list of processors'):
        _ = engine.get_sampler(['test1', 'test2'])


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_sampler_with_stream_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=True))
    sampler = engine.get_sampler(processor_id='tmp')
    results = sampler.run_sweep(
        program=_CIRCUIT, params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})]
    )
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client().run_job_over_stream.call_args[1]['project_id'] == 'proj'

    with pytest.raises(ValueError, match='list of processors'):
        _ = engine.get_sampler(['test1', 'test2'])


@mock.patch('cirq_google.cloud.quantum.QuantumEngineServiceClient')
def test_get_engine(build):
    # Default project id present.
    with mock.patch('google.auth.default', lambda *args, **kwargs: (None, 'project!')):
        eng = cirq_google.get_engine()
        assert eng.project_id == 'project!'

    # Nothing present.
    with mock.patch('google.auth.default', lambda *args, **kwargs: (None, None)):
        with pytest.raises(EnvironmentError, match='GOOGLE_CLOUD_PROJECT'):
            _ = cirq_google.get_engine()
        _ = cirq_google.get_engine('project!')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor')
def test_get_engine_device(get_processor):
    device_spec = util.pack_any(
        Merge(
            """
valid_qubits: "0_0"
valid_qubits: "1_1"
valid_qubits: "2_2"
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "1_1"
  }
}
valid_gates {
  gate_duration_picos: 1000
  cz {
  }
}
valid_gates {
  phased_xz {
  }
}
""",
            v2.device_pb2.DeviceSpecification(),
        )
    )

    get_processor.return_value = quantum.QuantumProcessor(device_spec=device_spec)
    device = cirq_google.get_engine_device('rainbow', 'project')
    assert device.metadata.qubit_set == frozenset(
        [cirq.GridQubit(0, 0), cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)]
    )
    device.validate_operation(cirq.X(cirq.GridQubit(2, 2)))
    device.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.GridQubit(1, 2)))
    with pytest.raises(ValueError):
        device.validate_operation(
            cirq.testing.DoesNotSupportSerializationGate()(cirq.GridQubit(0, 0))
        )
    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)))

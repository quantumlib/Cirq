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
import os
from unittest import mock
import numpy as np
import pytest

from google.protobuf import any_pb2
from google.protobuf.text_format import Merge

import cirq
import cirq.google as cg
from cirq.google.api import v1, v2
from cirq.google.engine.engine import EngineContext
from cirq.google.engine.client.quantum_v1alpha1 import types as qtypes

_CIRCUIT = cirq.Circuit(
    cirq.X(cirq.GridQubit(5, 2))**0.5,
    cirq.measure(cirq.GridQubit(5, 2), key='result'))


_CIRCUIT2 = cirq.Circuit(
    cirq.Y(cirq.GridQubit(5, 2))**0.5,
    cirq.measure(cirq.GridQubit(5, 2), key='result'))


def _to_any(proto):
    any_proto = qtypes.any_pb2.Any()
    any_proto.Pack(proto)
    return any_proto


def _to_timestamp(json_string):
    timestamp_proto = qtypes.timestamp_pb2.Timestamp()
    timestamp_proto.FromJsonString(json_string)
    return timestamp_proto


_A_RESULT = _to_any(
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
""", v1.program_pb2.Result()))

_RESULTS = _to_any(
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
""", v1.program_pb2.Result()))

_RESULTS_V2 = _to_any(
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
""", v2.result_pb2.Result()))

_BATCH_RESULTS_V2 = _to_any(
    Merge(
        """
results: [{
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
    }],
    },{
    sweep_results: [{
        repetitions: 1,
        parameterized_results: [{
            params: {
                assignments: {
                    key: 'a'
                    value: 3
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
                    value: 4
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
}]
""", v2.batch_pb2.BatchResult()))


@pytest.fixture(scope='session', autouse=True)
def mock_grpc_client():
    with mock.patch('cirq.google.engine.engine_client'
                    '.quantum.QuantumEngineServiceClient') as _fixture:
        yield _fixture


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_create_context(client):
    with pytest.raises(ValueError,
                       match='specify service_args and verbose or client'):
        EngineContext(cg.engine.engine.ProtoVersion.V1, {'args': 'test'}, True,
                      mock.Mock())
    with pytest.raises(ValueError, match='no longer supported'):
        _ = EngineContext(cg.engine.engine.ProtoVersion.V1, {'args': 'test'},
                          True)

    context = EngineContext(cg.engine.engine.ProtoVersion.V2, {'args': 'test'},
                            True)
    assert context.proto_version == cg.engine.engine.ProtoVersion.V2
    assert client.called_with({'args': 'test'}, True)

    assert context.copy().proto_version == context.proto_version
    assert context.copy().client == context.client
    assert context.copy() == context


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_create_engine(client):
    with pytest.raises(
            ValueError,
            match='provide context or proto_version, service_args and verbose'):
        cg.Engine('proj',
                  proto_version=cg.engine.engine.ProtoVersion.V2,
                  service_args={'args': 'test'},
                  verbose=True,
                  context=mock.Mock())

    assert cg.Engine(
        'proj',
        proto_version=cg.engine.engine.ProtoVersion.V2,
        service_args={
            'args': 'test'
        },
        verbose=True).context.proto_version == cg.engine.engine.ProtoVersion.V2
    assert client.called_with({'args': 'test'}, True)


def test_engine_str():
    engine = cg.Engine('proj',
                       proto_version=cg.engine.engine.ProtoVersion.V2,
                       service_args={'args': 'test'},
                       verbose=True)
    assert str(engine) == 'Engine(project_id=\'proj\')'


def setup_run_circuit_with_result_(client, result):
    client().create_program.return_value = (
        'prog', qtypes.QuantumProgram(name='projects/proj/programs/prog'))
    client().create_job.return_value = (
        'job-id',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job-id',
                          execution_status={'state': 'READY'}))
    client().get_job.return_value = qtypes.QuantumJob(
        execution_status={'state': 'SUCCESS'})
    client().get_job_results.return_value = qtypes.QuantumResult(result=result)


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_circuit(client):
    setup_run_circuit_with_result_(client, _A_RESULT)

    engine = cg.Engine(project_id='proj', service_args={'client_info': 1})
    result = engine.run(program=_CIRCUIT,
                        program_id='prog',
                        job_id='job-id',
                        processor_ids=['mysim'],
                        gate_set=cg.XMON)

    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    client.assert_called_with(service_args={'client_info': 1}, verbose=None)
    client.create_program.called_once_with()
    client.create_job.called_once_with(
        'projects/project-id/programs/test',
        qtypes.QuantumJob(
            name='projects/project-id/programs/test/jobs/job-id',
            scheduling_config={
                'priority': 50,
                'processor_selector': {
                    'processor_names': ['projects/project-id/processors/mysim']
                }
            },
            run_context=_to_any(
                v2.run_context_pb2.RunContext(parameter_sweeps=[
                    v2.run_context_pb2.ParameterSweep(repetitions=1)
                ]))), False)

    client.get_job.called_once_with('proj', 'prog')
    client.get_job_result.called_once_with()


def test_circuit_device_validation_fails():
    circuit = cirq.Circuit(device=cg.Foxtail)

    # Purposefully create an invalid Circuit by fiddling with internal bits.
    # This simulates a failure in the incremental checks.
    circuit._moments.append(cirq.Moment([
        cirq.Z(cirq.NamedQubit("dorothy"))]))
    engine = cg.Engine(project_id='project-id')
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        engine.run_sweep(program=circuit, gate_set=cg.XMON)
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        engine.create_program(circuit, gate_set=cg.XMON)


def test_no_gate_set():
    circuit = cirq.Circuit(device=cg.Sycamore)
    engine = cg.Engine(project_id='project-id')
    with pytest.raises(ValueError, match='No gate set'):
        engine.run(program=circuit)
    with pytest.raises(ValueError, match='No gate set'):
        engine.run_sweep(program=circuit)
    with pytest.raises(ValueError, match='No gate set'):
        engine.create_program(program=circuit)


def test_unsupported_program_type():
    engine = cg.Engine(project_id='project-id')
    with pytest.raises(TypeError, match='program'):
        engine.run(program="this isn't even the right type of thing!",
                   gate_set=cg.XMON)


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_circuit_failed(client):
    client().create_program.return_value = (
        'prog', qtypes.QuantumProgram(name='projects/proj/programs/prog'))
    client().create_job.return_value = (
        'job-id',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job-id',
                          execution_status={'state': 'READY'}))
    client().get_job.return_value = qtypes.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'FAILURE',
            'processor_name': 'myqc',
            'failure': {
                'error_code': 'SYSTEM_ERROR',
                'error_message': 'Not good'
            }
        })

    engine = cg.Engine(project_id='proj')
    with pytest.raises(
            RuntimeError,
            match='Job projects/proj/programs/prog/jobs/job-id on processor'
            ' myqc failed. SYSTEM_ERROR: Not good'):
        engine.run(program=_CIRCUIT, gate_set=cg.XMON)


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_circuit_failed_missing_processor_name(client):
    client().create_program.return_value = (
        'prog', qtypes.QuantumProgram(name='projects/proj/programs/prog'))
    client().create_job.return_value = (
        'job-id',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job-id',
                          execution_status={'state': 'READY'}))
    client().get_job.return_value = qtypes.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'FAILURE',
            'failure': {
                'error_code': 'SYSTEM_ERROR',
                'error_message': 'Not good'
            }
        })

    engine = cg.Engine(project_id='proj')
    with pytest.raises(
            RuntimeError,
            match='Job projects/proj/programs/prog/jobs/job-id on processor'
            ' UNKNOWN failed. SYSTEM_ERROR: Not good'):
        engine.run(program=_CIRCUIT, gate_set=cg.XMON)


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_circuit_cancelled(client):
    client().create_program.return_value = (
        'prog', qtypes.QuantumProgram(name='projects/proj/programs/prog'))
    client().create_job.return_value = (
        'job-id',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job-id',
                          execution_status={'state': 'READY'}))
    client().get_job.return_value = qtypes.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'CANCELLED',
        })

    engine = cg.Engine(project_id='proj')
    with pytest.raises(RuntimeError,
                       match='Job projects/proj/programs/prog/jobs/job-id'
                       ' failed in state CANCELLED.'):
        engine.run(program=_CIRCUIT, gate_set=cg.XMON)


@mock.patch('cirq.google.engine.engine_client.EngineClient')
@mock.patch('time.sleep', return_value=None)
def test_run_circuit_timeout(patched_time_sleep, client):
    client().create_program.return_value = (
        'prog', qtypes.QuantumProgram(name='projects/proj/programs/prog'))
    client().create_job.return_value = (
        'job-id',
        qtypes.QuantumJob(name='projects/proj/programs/prog/jobs/job-id',
                          execution_status={'state': 'READY'}))
    client().get_job.return_value = qtypes.QuantumJob(
        name='projects/proj/programs/prog/jobs/job-id',
        execution_status={
            'state': 'RUNNING',
        })

    engine = cg.Engine(project_id='project-id', timeout=600)
    with pytest.raises(RuntimeError, match='Timed out'):
        engine.run(program=_CIRCUIT, gate_set=cg.XMON)


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_sweep_params(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj')
    job = engine.run_sweep(
        program=_CIRCUIT,
        params=[cirq.ParamResolver({'a': 1}),
                cirq.ParamResolver({'a': 2})],
        gate_set=cg.XMON)
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}

    client().create_program.assert_called_once()
    client().create_job.assert_called_once()

    run_context = v2.run_context_pb2.RunContext()
    client().create_job.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 2
    for i, v in enumerate([1.0, 2.0]):
        assert sweeps[i].repetitions == 1
        assert sweeps[i].sweep.sweep_function.sweeps[
            0].single_sweep.points.points == [v]
    client().get_job.assert_called_once()
    client().get_job_results.assert_called_once()


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_multiple_times(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj',
                       proto_version=cg.engine.engine.ProtoVersion.V2)
    program = engine.create_program(program=_CIRCUIT, gate_set=cg.XMON)
    program.run(param_resolver=cirq.ParamResolver({'a': 1}))
    run_context = v2.run_context_pb2.RunContext()
    client().create_job.call_args[1]['run_context'].Unpack(run_context)
    sweeps1 = run_context.parameter_sweeps
    job2 = program.run_sweep(repetitions=2, params=cirq.Points('a', [3, 4]))
    client().create_job.call_args[1]['run_context'].Unpack(run_context)
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
    points1 = sweeps1[0].sweep.sweep_function.sweeps[0].single_sweep.points
    assert points1.points == [1]
    assert len(sweeps2) == 1
    assert sweeps2[0].repetitions == 2
    assert sweeps2[0].sweep.single_sweep.points.points == [3, 4]
    assert client().get_job.call_count == 2
    assert client().get_job_results.call_count == 2


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_sweep_v2(client):
    setup_run_circuit_with_result_(client, _RESULTS_V2)

    engine = cg.Engine(
        project_id='proj',
        proto_version=cg.engine.engine.ProtoVersion.V2,
    )
    job = engine.run_sweep(program=_CIRCUIT,
                           job_id='job-id',
                           params=cirq.Points('a', [1, 2]),
                           gate_set=cg.XMON)
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    client().create_program.assert_called_once()
    client().create_job.assert_called_once()
    run_context = v2.run_context_pb2.RunContext()
    client().create_job.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 1
    assert sweeps[0].repetitions == 1
    assert sweeps[0].sweep.single_sweep.points.points == [1, 2]
    client().get_job.assert_called_once()
    client().get_job_results.assert_called_once()


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_run_batch(client):
    setup_run_circuit_with_result_(client, _BATCH_RESULTS_V2)

    engine = cg.Engine(
        project_id='proj',
        proto_version=cg.engine.engine.ProtoVersion.V2,
    )
    job = engine.run_batch(
        gate_set=cg.XMON,
        programs=[_CIRCUIT, _CIRCUIT2],
        job_id='job-id',
        params_list=[cirq.Points('a', [1, 2]),
                     cirq.Points('a', [3, 4])],
        processor_ids=['mysim'])
    results = job.results()
    assert len(results) == 4
    for i, v in enumerate([1, 2, 3, 4]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    client().create_program.assert_called_once()
    client().create_job.assert_called_once()
    run_context = v2.batch_pb2.BatchRunContext()
    client().create_job.call_args[1]['run_context'].Unpack(run_context)
    assert len(run_context.run_contexts) == 2
    for idx, rc in enumerate(run_context.run_contexts):
        sweeps = rc.parameter_sweeps
        assert len(sweeps) == 1
        assert sweeps[0].repetitions == 1
        if idx == 0:
            assert sweeps[0].sweep.single_sweep.points.points == [1.0, 2.0]
        if idx == 1:
            assert sweeps[0].sweep.single_sweep.points.points == [3.0, 4.0]
    client().get_job.assert_called_once()
    client().get_job_results.assert_called_once()


def test_batch_size_validation_fails():
    engine = cg.Engine(
        project_id='proj',
        proto_version=cg.engine.engine.ProtoVersion.V2,
    )

    with pytest.raises(ValueError, match='Number of circuits and sweeps'):
        _ = engine.run_batch(programs=[_CIRCUIT, _CIRCUIT2],
                             gate_set=cg.XMON,
                             job_id='job-id',
                             params_list=[
                                 cirq.Points('a', [1, 2]),
                                 cirq.Points('a', [3, 4]),
                                 cirq.Points('a', [5, 6])
                             ],
                             processor_ids=['mysim'])

    with pytest.raises(ValueError, match='Processor id must be specified'):
        _ = engine.run_batch(
            programs=[_CIRCUIT, _CIRCUIT2],
            gate_set=cg.XMON,
            job_id='job-id',
            params_list=[cirq.Points('a', [1, 2]),
                         cirq.Points('a', [3, 4])])

    with pytest.raises(ValueError, match='Gate set must be specified'):
        _ = engine.run_batch(
            programs=[_CIRCUIT, _CIRCUIT2],
            job_id='job-id',
            params_list=[cirq.Points('a', [1, 2]),
                         cirq.Points('a', [3, 4])],
            processor_ids=['mysim'])


def test_bad_sweep_proto():
    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.ProtoVersion.UNDEFINED)
    program = cg.EngineProgram('proj', 'prog', engine.context)
    with pytest.raises(ValueError, match='invalid run context proto version'):
        program.run_sweep()


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_bad_result_proto(client):
    result = any_pb2.Any()
    result.CopyFrom(_RESULTS_V2)
    result.type_url = 'type.googleapis.com/unknown'
    setup_run_circuit_with_result_(client, result)

    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.engine.engine.ProtoVersion.V2)
    job = engine.run_sweep(program=_CIRCUIT,
                           job_id='job-id',
                           params=cirq.Points('a', [1, 2]),
                           gate_set=cg.XMON)
    with pytest.raises(ValueError, match='invalid result proto version'):
        job.results()


def test_bad_program_proto():
    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.engine.engine.ProtoVersion.UNDEFINED)
    with pytest.raises(ValueError, match='invalid program proto version'):
        engine.run_sweep(program=_CIRCUIT, gate_set=cg.XMON)
    with pytest.raises(ValueError, match='invalid program proto version'):
        engine.create_program(_CIRCUIT, gate_set=cg.XMON)


def test_get_program():
    assert cg.Engine(project_id='proj').get_program('prog').program_id == 'prog'


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_create_program(client):
    client().create_program.return_value = ('prog', qtypes.QuantumProgram())
    result = cg.Engine(project_id='proj').create_program(_CIRCUIT,
                                                         'prog',
                                                         gate_set=cg.XMON)
    client().create_program.assert_called_once()
    assert result.program_id == 'prog'


@mock.patch('cirq.google.engine.engine_client.EngineClient.list_processors')
def test_list_processors(list_processors):
    processor1 = qtypes.QuantumProcessor(
        name='projects/proj/processors/xmonsim')
    processor2 = qtypes.QuantumProcessor(
        name='projects/proj/processors/gmonsim')
    list_processors.return_value = [processor1, processor2]

    result = cg.Engine(project_id='proj').list_processors()
    list_processors.assert_called_once_with('proj')
    assert [p.processor_id for p in result] == ['xmonsim', 'gmonsim']


def test_get_processor():
    assert cg.Engine(
        project_id='proj').get_processor('xmonsim').processor_id == 'xmonsim'


@mock.patch('cirq.google.engine.engine_client.EngineClient')
def test_sampler(client):
    setup_run_circuit_with_result_(client, _RESULTS)

    engine = cg.Engine(project_id='proj')
    sampler = engine.sampler(processor_id='tmp', gate_set=cg.XMON)
    results = sampler.run_sweep(
        program=_CIRCUIT,
        params=[cirq.ParamResolver({'a': 1}),
                cirq.ParamResolver({'a': 2})])
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client().create_program.call_args[0][0] == 'proj'


@mock.patch('cirq.google.engine.client.quantum.QuantumEngineServiceClient')
def test_get_engine(build):
    # Default project id present.
    with mock.patch.dict(os.environ, {
            'GOOGLE_CLOUD_PROJECT': 'project!',
    },
                         clear=True):
        eng = cirq.google.get_engine()
        assert eng.project_id == 'project!'

    # Nothing present.
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError, match='GOOGLE_CLOUD_PROJECT'):
            _ = cirq.google.get_engine()
        _ = cirq.google.get_engine('project!')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_processor')
def test_get_engine_device(get_processor):
    device_spec = _to_any(
        Merge(
            """
valid_gate_sets: [{
    name: 'test_set',
    valid_gates: [{
        id: 'x',
        number_of_qubits: 1,
        gate_duration_picos: 1000,
        valid_targets: ['1q_targets']
    }]
}],
valid_qubits: ['0_0', '1_1'],
valid_targets: [{
    name: '1q_targets',
    target_ordering: SYMMETRIC,
    targets: [{
        ids: ['0_0']
    }]
}]
""", v2.device_pb2.DeviceSpecification()))

    gate_set = cg.SerializableGateSet(
        gate_set_name='x_gate_set',
        serializers=[
            cg.GateOpSerializer(gate_type=cirq.XPowGate,
                                serialized_gate_id='x',
                                args=[])
        ],
        deserializers=[
            cg.GateOpDeserializer(serialized_gate_id='x',
                                  gate_constructor=cirq.XPowGate,
                                  args=[])
        ],
    )

    get_processor.return_value = qtypes.QuantumProcessor(
        device_spec=device_spec)
    device = cirq.google.get_engine_device('rainbow',
                                           'project',
                                           gatesets=[gate_set])
    assert set(device.qubits) == {cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)}
    device.validate_operation(cirq.X(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.GridQubit(1, 2)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.Y(cirq.GridQubit(0, 0)))

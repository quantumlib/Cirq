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
from unittest import mock
import numpy as np
import pytest

from google.protobuf import any_pb2
from google.protobuf.text_format import Merge
from google.api_core import exceptions

import cirq
import cirq.google as cg
from cirq.google.engine.client import quantum
from cirq.google.engine.client.quantum_v1alpha1 import types as qtypes
from cirq.google.api import v1, v2

_CIRCUIT = cirq.Circuit(
    cirq.X(cirq.GridQubit(5, 2))**0.5,
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

_CALIBRATION = qtypes.QuantumCalibration(
    name='projects/foo/processors/xmonsim/calibrations/1562715599',
    timestamp=_to_timestamp('2019-07-09T23:39:59Z'),
    data=_to_any(
        Merge(
            """
    timestamp_ms: 1562544000021,
    metrics: [{
        name: 'xeb',
        targets: ['0_0', '0_1'],
        values: [{
            double_val: .9999
        }]
    }, {
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{
            double_val: .9998
        }]
    }, {
        name: 't1',
        targets: ['q0_0'],
        values: [{
            double_val: 321
        }]
    }, {
        name: 't1',
        targets: ['q0_1'],
        values: [{
            double_val: 911
        }]
    }, {
        name: 't1',
        targets: ['q1_0'],
        values: [{
            double_val: 505
        }]
    }, {
        name: 'globalMetric',
        values: [{
            int32_val: 12300
        }]
    }]
""", v2.metrics_pb2.MetricsSnapshot())))

_DEVICE_SPEC = _to_any(
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


def test_job_config_repr():
    v = cirq.google.JobConfig(job_id='my-job-id')
    cirq.testing.assert_equivalent_repr(v)


def setup_run_circuit_with_result_(client_constructor, result):
    client = mock.Mock()
    client_constructor.return_value = client

    client.create_quantum_program.return_value = qtypes.QuantumProgram(
        name='projects/project-id/programs/test')
    client.create_quantum_job.return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={'state': 'READY'})
    client.get_quantum_job.return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={'state': 'SUCCESS'})
    client.get_quantum_result.return_value = qtypes.QuantumResult(result=result)
    return client


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_circuit(client_constructor):
    client = setup_run_circuit_with_result_(client_constructor, _A_RESULT)

    engine = cg.Engine(project_id='project-id', service_args={'client_info': 1})
    result = engine.run(program=_CIRCUIT,
                        job_config=cg.JobConfig('job-id'),
                        processor_ids=['mysim'])

    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    client_constructor.assert_called_with(client_info=1)
    assert client.create_quantum_program.call_args[0][
        0] == 'projects/project-id'
    assert client.create_quantum_job.call_args[0] == (
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
                v1.program_pb2.RunContext(parameter_sweeps=[{
                    'repetitions': 1
                }]))), False)

    assert client.get_quantum_job.call_count == 1
    assert client.get_quantum_result.call_count == 1


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_circuit_device_validation_fails(client_constructor):
    circuit = cirq.Circuit(device=cg.Foxtail)

    # Purposefully create an invalid Circuit by fiddling with internal bits.
    # This simulates a failure in the incremental checks.
    circuit._moments.append(cirq.Moment([
        cirq.Z(cirq.NamedQubit("dorothy"))]))
    engine = cg.Engine(project_id='project-id')
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        engine.run_sweep(program=circuit)
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        engine.create_program(circuit)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_unsupported_program_type(client_constructor):
    engine = cg.Engine(project_id='project-id')
    with pytest.raises(TypeError, match='program'):
        engine.run(program="this isn't even the right type of thing!")


def setup_run_circuit_with_job_state_(client_constructor, job_return_value):
    client = mock.Mock()
    client_constructor.return_value = client

    client.create_quantum_program.return_value = qtypes.QuantumProgram(
        name='projects/project-id/programs/test')
    client.create_quantum_job.return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={'state': 'READY'})
    client.get_quantum_job.return_value = job_return_value
    return client


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_circuit_failed(client_constructor):
    job_return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={
            'state': 'FAILURE',
            'processor_name': 'myqc',
            'failure': {
                'error_code': 'SYSTEM_ERROR',
                'error_message': 'Not good'
            }
        })
    setup_run_circuit_with_job_state_(client_constructor, job_return_value)

    engine = cg.Engine(project_id='project-id')
    with pytest.raises(RuntimeError, match='myqc'):
        engine.run(program=_CIRCUIT)
    with pytest.raises(RuntimeError, match='SYSTEM_ERROR'):
        engine.run(program=_CIRCUIT)
    with pytest.raises(RuntimeError, match='Not good'):
        engine.run(program=_CIRCUIT)
    with pytest.raises(RuntimeError, match='jobs/test'):
        engine.run(program=_CIRCUIT)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_circuit_failed_missing_processor_name(client_constructor):
    job_return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={
            'state': 'FAILURE',
            'failure': {
                'error_code': 'SYSTEM_ERROR',
                'error_message': 'Not good'
            }
        })
    setup_run_circuit_with_job_state_(client_constructor, job_return_value)

    engine = cg.Engine(project_id='project-id')
    with pytest.raises(RuntimeError, match='UNKNOWN'):
        engine.run(program=_CIRCUIT)
    with pytest.raises(RuntimeError, match='SYSTEM_ERROR'):
        engine.run(program=_CIRCUIT)
    with pytest.raises(RuntimeError, match='Not good'):
        engine.run(program=_CIRCUIT)
    with pytest.raises(RuntimeError, match='jobs/test'):
        engine.run(program=_CIRCUIT)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_circuit_cancelled(client_constructor):
    job_return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={
            'state': 'CANCELLED',
        })
    setup_run_circuit_with_job_state_(client_constructor, job_return_value)

    engine = cg.Engine(project_id='project-id')
    with pytest.raises(RuntimeError, match='CANCELLED'):
        engine.run(program=_CIRCUIT)
    with pytest.raises(RuntimeError, match='jobs/test'):
        engine.run(program=_CIRCUIT)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
@mock.patch('time.sleep', return_value=None)
def test_run_circuit_timeout(client_constructor, patched_time_sleep):
    job_return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={
            'state': 'RUNNING',
        })
    setup_run_circuit_with_job_state_(client_constructor, job_return_value)

    engine = cg.Engine(project_id='project-id')
    with pytest.raises(RuntimeError, match='Timed out'):
        engine.run(program=_CIRCUIT)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_sweep_params(client_constructor):
    client = setup_run_circuit_with_result_(client_constructor, _RESULTS)

    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(
        program=_CIRCUIT,
        job_config=cg.JobConfig('project-id'),
        params=[cirq.ParamResolver({'a': 1}),
                cirq.ParamResolver({'a': 2})])
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}

    assert client.create_quantum_program.call_args[0][
        0] == 'projects/project-id'
    assert client.create_quantum_job.call_args[0][
        0] == 'projects/project-id/programs/test'
    run_context = v1.program_pb2.RunContext()
    client.create_quantum_job.call_args[0][1].run_context.Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 2
    for i, v in enumerate([1, 2]):
        assert sweeps[i].repetitions == 1
        assert sweeps[i].sweep.factors[0].sweeps[0].points.points == [v]
    assert client.get_quantum_job.call_count == 1
    assert client.get_quantum_result.call_count == 1


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_sweep_v1(client_constructor):
    client = setup_run_circuit_with_result_(client_constructor, _RESULTS)

    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(program=_CIRCUIT,
                           job_config=cg.JobConfig('project-id'),
                           params=cirq.Points('a', [1, 2]))
    results = job.results()
    assert engine.proto_version == cg.engine.engine.ProtoVersion.V1
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client.create_quantum_program.call_args[0][
        0] == 'projects/project-id'
    assert client.create_quantum_job.call_args[0][
        0] == 'projects/project-id/programs/test'
    run_context = v1.program_pb2.RunContext()
    client.create_quantum_job.call_args[0][1].run_context.Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 1
    assert sweeps[0].repetitions == 1
    assert sweeps[0].sweep.factors[0].sweeps[0].points.points == [1, 2]
    assert client.get_quantum_job.call_count == 1
    assert client.get_quantum_result.call_count == 1


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_multiple_times(client_constructor):
    client = setup_run_circuit_with_result_(client_constructor, _RESULTS)

    engine = cg.Engine(project_id='project-id')
    program = engine.create_program(program=_CIRCUIT)
    program.run(param_resolver=cirq.ParamResolver({'a': 1}))
    run_context = v1.program_pb2.RunContext()
    client.create_quantum_job.call_args[0][1].run_context.Unpack(run_context)
    sweeps1 = run_context.parameter_sweeps
    job2 = program.run_sweep(repetitions=2, params=cirq.Points('a', [3, 4]))
    client.create_quantum_job.call_args[0][1].run_context.Unpack(run_context)
    sweeps2 = run_context.parameter_sweeps
    results = job2.results()
    assert engine.proto_version == cg.engine.engine.ProtoVersion.V1
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert len(sweeps1) == 1
    assert sweeps1[0].repetitions == 1
    assert sweeps1[0].sweep.factors[0].sweeps[0].points.points == [1]
    assert len(sweeps2) == 1
    assert sweeps2[0].repetitions == 2
    assert sweeps2[0].sweep.factors[0].sweeps[0].points.points == [3, 4]
    assert client.get_quantum_job.call_count == 2
    assert client.get_quantum_result.call_count == 2


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_run_sweep_v2(client_constructor):
    client = setup_run_circuit_with_result_(client_constructor, _RESULTS_V2)

    engine = cg.Engine(
        project_id='project-id',
        proto_version=cg.engine.engine.ProtoVersion.V2,
    )
    job = engine.run_sweep(program=_CIRCUIT,
                           job_config=cg.JobConfig('project-id'),
                           params=cirq.Points('a', [1, 2]))
    results = job.results()
    assert engine.proto_version == cg.engine.engine.ProtoVersion.V2
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client.create_quantum_program.call_args[0][
        0] == 'projects/project-id'
    assert client.create_quantum_job.call_args[0][
        0] == 'projects/project-id/programs/test'
    run_context = v2.run_context_pb2.RunContext()
    client.create_quantum_job.call_args[0][1].run_context.Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 1
    assert sweeps[0].repetitions == 1
    assert sweeps[0].sweep.single_sweep.points.points == [1, 2]
    assert client.get_quantum_job.call_count == 1
    assert client.get_quantum_result.call_count == 1


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_bad_sweep_proto(client_constructor):
    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.ProtoVersion.UNDEFINED)
    program = cg.EngineProgram({'name': 'foo'}, engine)
    with pytest.raises(ValueError, match='invalid run context proto version'):
        program.run_sweep()


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_bad_result_proto(client_constructor):
    result = any_pb2.Any()
    result.CopyFrom(_RESULTS_V2)
    result.type_url = 'type.googleapis.com/unknown'
    setup_run_circuit_with_result_(client_constructor, result)

    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.engine.engine.ProtoVersion.V2)
    job = engine.run_sweep(program=_CIRCUIT,
                           job_config=cg.JobConfig('project-id'),
                           params=cirq.Points('a', [1, 2]))
    with pytest.raises(ValueError, match='invalid result proto version'):
        job.results()


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_bad_program_proto(client_constructor):
    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.engine.engine.ProtoVersion.UNDEFINED)
    with pytest.raises(ValueError, match='invalid program proto version'):
        engine.run_sweep(program=_CIRCUIT)
    with pytest.raises(ValueError, match='invalid program proto version'):
        engine.create_program(_CIRCUIT)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_bad_priority(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client

    client.create_quantum_program.return_value = qtypes.QuantumProgram(
        name='projects/project-id/programs/test')
    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.engine.engine.ProtoVersion.V2)
    with pytest.raises(ValueError, match='priority must be'):
        engine.run(program=_CIRCUIT, priority=1001)


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_cancel(client_constructor):
    job_return_value = qtypes.QuantumJob(
        name='projects/project-id/programs/test/jobs/test',
        execution_status={
            'state': 'CANCELLED',
        })
    client = setup_run_circuit_with_job_state_(client_constructor,
                                               job_return_value)

    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(program=_CIRCUIT,
                           job_config=cg.JobConfig('project-id'))
    job.cancel()
    assert job.job_resource_name == ('projects/project-id/programs/test/'
                                     'jobs/test')
    assert job.status() == 'CANCELLED'
    assert client.cancel_quantum_job.call_args[0][
        0] == 'projects/project-id/programs/test/jobs/test'


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_program_labels(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client

    program_name = 'projects/my-proj/programs/my-prog'
    engine = cg.Engine(project_id='project-id')

    client.get_quantum_program.return_value = qtypes.QuantumProgram(labels={
        'a': '1',
        'b': '1'
    })
    engine.add_program_labels(program_name, {'b': '1'})
    engine.remove_program_labels(program_name, ['c'])

    assert client.update_quantum_program.call_count == 0

    def update():
        return client.update_quantum_program.call_args[0][1]

    client.get_quantum_program.return_value = qtypes.QuantumProgram(labels={
        'a': '1',
        'b': '1'
    })
    engine.add_program_labels(program_name, {'a': '2', 'c': '1'})

    assert update().labels == {'a': '2', 'b': '1', 'c': '1'}
    assert update().label_fingerprint == ''

    client.get_quantum_program.return_value = qtypes.QuantumProgram(
        labels={
            'a': '1',
            'b': '1'
        }, label_fingerprint='abcdef')
    engine.set_program_labels(program_name, {'s': '1', 'p': '1'})
    assert update().labels == {'s': '1', 'p': '1'}
    assert update().label_fingerprint == 'abcdef'

    client.get_quantum_program.return_value = qtypes.QuantumProgram(
        labels={
            'a': '1',
            'b': '1'
        }, label_fingerprint='abcdef')
    engine.remove_program_labels(program_name, ['a', 'c'])
    assert update().labels == {'b': '1'}
    assert update().label_fingerprint == 'abcdef'


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_job_labels(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client

    job_name = 'projects/my-proj/programs/my-prog/jobs/my-job'
    engine = cg.Engine(project_id='project-id')

    client.get_quantum_job.return_value = qtypes.QuantumJob(labels={
        'a': '1',
        'b': '1'
    })
    engine.add_job_labels(job_name, {'b': '1'})
    engine.remove_job_labels(job_name, ['c'])

    assert client.update_quantum_job.call_count == 0

    def update():
        return client.update_quantum_job.call_args[0][1]

    client.get_quantum_job.return_value = qtypes.QuantumJob(labels={
        'a': '1',
        'b': '1'
    })
    engine.add_job_labels(job_name, {'a': '2', 'c': '1'})

    assert update().labels == {'a': '2', 'b': '1', 'c': '1'}
    assert update().label_fingerprint == ''

    client.get_quantum_job.return_value = qtypes.QuantumJob(
        labels={
            'a': '1',
            'b': '1'
        }, label_fingerprint='abcdef')
    engine.set_job_labels(job_name, {'s': '1', 'p': '1'})
    assert update().labels == {'s': '1', 'p': '1'}
    assert update().label_fingerprint == 'abcdef'

    client.get_quantum_job.return_value = qtypes.QuantumJob(
        labels={
            'a': '1',
            'b': '1'
        }, label_fingerprint='abcdef')
    engine.remove_job_labels(job_name, ['a', 'c'])
    assert update().labels == {'b': '1'}
    assert update().label_fingerprint == 'abcdef'


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_implied_job_config(client_constructor):
    eng = cg.Engine(project_id='project_id')

    # Infer all from project id.
    implied = eng.implied_job_config(cg.JobConfig())
    assert implied.job_id.startswith('job-')
    assert len(implied.job_id) == 26

    # Force all.
    implied = eng.implied_job_config(cg.JobConfig(job_id='c'))
    assert implied.job_id == 'c'


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_get_program(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client

    fake_result = qtypes.QuantumProgram(name='project/my-project/program/foo')
    client.get_quantum_program.return_value = fake_result
    result = cg.Engine(project_id='my-project').get_program('foo')
    assert client.get_quantum_program.call_args[0][
        0] == 'projects/my-project/programs/foo'
    assert result == fake_result


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_create_program(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client

    fake_result = qtypes.QuantumProgram(name='project/my-project/program/foo')
    client.create_quantum_program.return_value = fake_result
    result = cg.Engine(project_id='my-project').create_program(_CIRCUIT, 'foo')
    assert client.create_quantum_program.call_args[0][
        1].name == 'projects/my-project/programs/foo'
    assert result.resource_name == fake_result.name


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_list_processors(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    PROCESSOR1 = qtypes.QuantumProcessor(
        name='projects/myproject/processors/xmonsim'),
    PROCESSOR2 = qtypes.QuantumProcessor(
        name='projects/myproject/processors/gmonsin'),
    client.list_quantum_processors.return_value = [PROCESSOR1, PROCESSOR2]

    result = cg.Engine(project_id='myproject').list_processors()
    assert client.list_quantum_processors.call_args[0][
        0] == 'projects/myproject'
    assert result == [PROCESSOR1, PROCESSOR2]


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_latest_calibration(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    client.get_quantum_calibration.return_value = _CALIBRATION
    calibration = cg.Engine(project_id='myproject').get_latest_calibration('x')
    assert client.get_quantum_calibration.call_args[0][
        0] == 'projects/myproject/processors/x/calibrations/current'
    assert calibration.timestamp == 1562544000021
    assert set(calibration.keys()) == set(['xeb', 't1', 'globalMetric'])


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_latest_calibration_error(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    client.get_quantum_calibration.side_effect =\
        exceptions.BadGateway('x-error')
    with pytest.raises(cg.engine.engine.EngineException, match='x-error'):
        cg.Engine(project_id='myproject').get_latest_calibration('x')


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_missing_latest_calibration(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    client.get_quantum_calibration.side_effect = exceptions.NotFound('')
    calibration = cg.Engine(project_id='myproject').get_latest_calibration('x')
    assert client.get_quantum_calibration.call_args[0][
        0] == 'projects/myproject/processors/x/calibrations/current'
    assert not calibration


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_calibration_from_job(client_constructor):
    calibration_name = '/project/p/processor/x/calibrations/123'
    client = setup_run_circuit_with_job_state_(
        client_constructor,
        qtypes.QuantumJob(name='projects/project-id/programs/test/jobs/test',
                          execution_status={
                              'state': 'SUCCESS',
                              'calibration_name': calibration_name
                          }))
    client.get_quantum_calibration.return_value = _CALIBRATION

    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(program=_CIRCUIT)

    job.status()
    calibration = job.get_calibration()
    assert calibration.timestamp == 1562544000021
    assert set(calibration.keys()) == set(['xeb', 't1', 'globalMetric'])
    assert client.get_quantum_calibration.call_args[0][0] == calibration_name


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_device_specification(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    client.get_quantum_processor.return_value = qtypes.QuantumProcessor(
        device_spec=_DEVICE_SPEC)
    device_spec = cg.Engine(
        project_id='myproject').get_device_specification('x')
    assert client.get_quantum_processor.call_args[0][
        0] == 'projects/myproject/processors/x'

    # Construct expected device proto based on example
    expected = v2.device_pb2.DeviceSpecification()
    gs = expected.valid_gate_sets.add()
    gs.name = 'test_set'
    gates = gs.valid_gates.add()
    gates.id = 'x'
    gates.number_of_qubits = 1
    gates.gate_duration_picos = 1000
    gates.valid_targets.extend(['1q_targets'])
    expected.valid_qubits.extend(['0_0', '1_1'])
    target = expected.valid_targets.add()
    target.name = '1q_targets'
    target.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = target.targets.add()
    new_target.ids.extend(['0_0'])

    assert device_spec == expected


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_missing_device_specification(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    client.get_quantum_processor.return_value = qtypes.QuantumProcessor()
    device_spec = cg.Engine(
        project_id='myproject').get_device_specification('x')
    assert client.get_quantum_processor.call_args[0][
        0] == 'projects/myproject/processors/x'
    assert device_spec == v2.device_pb2.DeviceSpecification()


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_sampler(client_constructor):
    client = setup_run_circuit_with_result_(client_constructor, _RESULTS)

    engine = cg.Engine(project_id='project-id')
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
    assert client.create_quantum_program.call_args[0][
        0] == 'projects/project-id'


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_api_doesnt_retry_not_found_errors(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    client.get_quantum_program.side_effect = exceptions.NotFound('not found')

    engine = cg.Engine(project_id='project-id')
    with pytest.raises(cg.engine.engine.EngineException, match='not found'):
        engine.get_program('foo')
    assert client.get_quantum_program.call_count == 1


@mock.patch.object(quantum, 'QuantumEngineServiceClient', autospec=True)
def test_api_retry_5xx_errors(client_constructor):
    client = mock.Mock()
    client_constructor.return_value = client
    client.get_quantum_program.side_effect = exceptions.ServiceUnavailable(
        'internal error')

    engine = cg.Engine(project_id='project-id')
    with pytest.raises(TimeoutError,
                       match='Reached max retry attempts.*internal error'):
        engine.max_retry_delay = 1  # 1 second
        engine.get_program('foo')
    assert client.get_quantum_program.call_count > 1

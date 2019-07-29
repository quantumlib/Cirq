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
import base64
import re
from unittest import mock
import numpy as np
import pytest

from apiclient import discovery, http

import cirq
import cirq.google as cg


_A_RESULT = {
    '@type':
    'type.googleapis.com/cirq.api.google.v1.Result',
    'sweepResults': [{
        'repetitions':
        1,
        'measurementKeys': [{
            'key': 'q',
            'qubits': [{
                'row': 1,
                'col': 1
            }]
        }],
        'parameterizedResults': [{
            'params': {
                'assignments': {
                    'a': 1
                }
            },
            'measurementResults': base64.b64encode(b'01')
        }]
    }]
}

_RESULTS = {
    '@type':
    'type.googleapis.com/cirq.api.google.v1.Result',
    'sweepResults': [{
        'repetitions':
        1,
        'measurementKeys': [{
            'key': 'q',
            'qubits': [{
                'row': 1,
                'col': 1
            }]
        }],
        'parameterizedResults': [{
            'params': {
                'assignments': {
                    'a': 1
                }
            },
            'measurementResults': base64.b64encode(b'01')
        }, {
            'params': {
                'assignments': {
                    'a': 2
                }
            },
            'measurementResults': base64.b64encode(b'01')
        }]
    }]
}

_RESULTS_V2 = {
    '@type':
    'type.googleapis.com/cirq.api.google.v2.Result',
    'sweepResults': [
        {
            'repetitions':
            1,
            'parameterizedResults': [
                {
                    'params': {
                        'assignments': {
                            'a': 1
                        }
                    },
                    'measurementResults': [
                        {
                            'key':
                            'q',
                            'qubitMeasurementResults': [
                                {
                                    'qubit': {
                                        'id': '1_1'
                                    },
                                    'results': base64.b64encode(b'01'),
                                },
                            ],
                        },
                    ],
                },
                {
                    'params': {
                        'assignments': {
                            'a': 2
                        }
                    },
                    'measurementResults': [
                        {
                            'key':
                            'q',
                            'qubitMeasurementResults': [
                                {
                                    'qubit': {
                                        'id': '1_1'
                                    },
                                    'results': base64.b64encode(b'01'),
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ],
}


_CALIBRATION = {
    'name': 'projects/foo/processors/xmonsim/calibrations/1562715599',
    'timestamp': '2019-07-09T23:39:59Z',
    'data': {
        '@type':
        'type.googleapis.com/cirq.api.google.v2.MetricsSnapshot',
        'timestampMs':
        '1562544000021',
        'metrics': [{
            'name': 'xeb',
            'targets': ['q0_0', 'q0_1'],
            'values': [{
                'doubleVal': .9999
            }]
        }, {
            'name': 'xeb',
            'targets': ['q0_0', 'q1_0'],
            'values': [{
                'doubleVal': .9998
            }]
        }, {
            'name': 't1',
            'targets': ['q0_0'],
            'values': [{
                'doubleVal': 321
            }]
        }, {
            'name': 't1',
            'targets': ['q0_1'],
            'values': [{
                'doubleVal': 911
            }]
        }, {
            'name': 't1',
            'targets': ['q1_0'],
            'values': [{
                'doubleVal': 505
            }]
        }]
    }
}


def test_repr():
    v = cirq.google.JobConfig(program_id='my-program-id', job_id='my-job-id')
    cirq.testing.assert_equivalent_repr(v)


@mock.patch.object(discovery, 'build')
def test_run_circuit(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'}
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'READY'}}
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'SUCCESS'}}
    jobs.getResult().execute.return_value = {
        'result': _A_RESULT}

    result = cg.Engine(project_id='project-id').run(
        program=cirq.Circuit(),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'))
    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum',
                             'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}'),
                             requestBuilder=mock.ANY)
    assert programs.create.call_args[1]['parent'] == 'projects/project-id'
    assert jobs.create.call_args[1][
        'parent'] == 'projects/project-id/programs/test'
    assert jobs.get().execute.call_count == 1
    assert jobs.getResult().execute.call_count == 1


@mock.patch.object(discovery, 'build')
def test_circuit_device_validation_fails(build):
    circuit = cirq.Circuit(device=cg.Foxtail)

    # Purposefully create an invalid Circuit by fiddling with internal bits.
    # This simulates a failure in the incremental checks.
    circuit._moments.append(cirq.Moment([
        cirq.Z(cirq.NamedQubit("dorothy"))]))

    with pytest.raises(ValueError, match='Unsupported qubit type'):
        cg.Engine(project_id='project-id').run(
            program=circuit, job_config=cg.JobConfig('project-id'))


@mock.patch.object(discovery, 'build')
def test_schedule_device_validation_fails(build):
    scheduled_op = cirq.ScheduledOperation(time=None,
                                           duration=cirq.Duration(),
                                           operation=cirq.H.on(
                                               cirq.NamedQubit("dorothy")))
    schedule = cirq.Schedule(device=cg.Foxtail,
                             scheduled_operations=[scheduled_op])

    with pytest.raises(ValueError):
        cg.Engine(project_id='project-id').run(program=schedule)


@mock.patch.object(discovery, 'build')
def test_circuit_device_validation_passes_non_xmon_gate(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'}
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'READY'}}
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'SUCCESS'}}
    jobs.getResult().execute.return_value = {
        'result': _A_RESULT}

    circuit = cirq.Circuit.from_ops(cirq.H.on(cirq.GridQubit(0, 1)),
                                    device=cg.Foxtail)
    result = cg.Engine(project_id='project-id').run(
        program=circuit, job_config=cg.JobConfig('project-id'))
    assert result.repetitions == 1


@mock.patch.object(discovery, 'build')
def test_unsupported_program_type(build):
    eng = cg.Engine(project_id='project-id')
    with pytest.raises(TypeError, match='program'):
        eng.run(program="this isn't even the right type of thing!",
                job_config=cg.JobConfig('project-id'))


@mock.patch.object(discovery, 'build')
def test_run_circuit_failed(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'}
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'READY'}}
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'FAILURE'}}

    with pytest.raises(RuntimeError, match='It is in state FAILURE'):
        cg.Engine(project_id='project-id').run(
            program=cirq.Circuit(),
            job_config=cg.JobConfig('project-id',
                                    gcs_prefix='gs://bucket/folder'))


@mock.patch.object(discovery, 'build')
def test_default_prefix(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'}
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'READY'}}
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'SUCCESS'}}
    jobs.getResult().execute.return_value = {
        'result': _A_RESULT}

    result = cg.Engine(project_id='project-id').run(
        program=cirq.Circuit(), job_config=cg.JobConfig('org.com:project-id'))
    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum',
                             'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}'),
                             requestBuilder=mock.ANY)
    assert programs.create.call_args[1]['body']['gcs_code_location'][
        'uri'].startswith('gs://gqe-project-id/programs/')


@mock.patch.object(discovery, 'build')
def test_run_sweep_params(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'}
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'READY'}}
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'SUCCESS'}}
    jobs.getResult().execute.return_value = {
        'result': _RESULTS}

    job = cg.Engine(project_id='project-id').run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'),
        params=[cirq.ParamResolver({'a': 1}),
                cirq.ParamResolver({'a': 2})])
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum',
                             'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}'),
                             requestBuilder=mock.ANY)
    assert programs.create.call_args[1]['parent'] == 'projects/project-id'
    sweeps = jobs.create.call_args[1]['body']['run_context']['parameter_sweeps']
    assert len(sweeps) == 2
    for i, v in enumerate([1, 2]):
        assert sweeps[i]['repetitions'] == 1
        assert sweeps[i]['sweep']['factors'][0]['sweeps'][0]['points'][
            'points'] == [v]
    assert jobs.create.call_args[1][
        'parent'] == 'projects/project-id/programs/test'
    assert jobs.get().execute.call_count == 1
    assert jobs.getResult().execute.call_count == 1

@mock.patch.object(discovery, 'build')
def test_run_sweep_v1(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'}
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'READY'}}
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'SUCCESS'}}
    jobs.getResult().execute.return_value = {
        'result': _RESULTS}

    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'),
        params=cirq.Points('a', [1, 2]))
    results = job.results()
    assert engine.proto_version == cg.engine.engine.ProtoVersion.V1
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum',
                             'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}'),
                             requestBuilder=mock.ANY)
    assert programs.create.call_args[1]['parent'] == 'projects/project-id'
    sweeps = jobs.create.call_args[1]['body']['run_context']['parameter_sweeps']
    assert len(sweeps) == 1
    assert sweeps[0]['repetitions'] == 1
    assert sweeps[0]['sweep']['factors'][0]['sweeps'][0]['points'][
        'points'] == [1, 2]
    assert jobs.create.call_args[1][
        'parent'] == 'projects/project-id/programs/test'
    assert jobs.get().execute.call_count == 1
    assert jobs.getResult().execute.call_count == 1


@mock.patch.object(discovery, 'build')
def test_run_sweep_v2(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'
    }
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {
            'state': 'READY'
        }
    }
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {
            'state': 'SUCCESS'
        }
    }
    jobs.getResult().execute.return_value = {'result': _RESULTS_V2}

    engine = cg.Engine(
        project_id='project-id',
        proto_version=cg.engine.engine.ProtoVersion.V2,
    )
    job = engine.run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'),
        params=cirq.Points('a', [1, 2]))
    results = job.results()
    assert engine.proto_version == cg.engine.engine.ProtoVersion.V2
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum',
                             'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}'),
                             requestBuilder=mock.ANY)
    assert programs.create.call_args[1]['parent'] == 'projects/project-id'
    sweeps = jobs.create.call_args[1]['body']['run_context']['parameterSweeps']
    assert len(sweeps) == 1
    assert sweeps[0]['repetitions'] == 1
    assert sweeps[0]['sweep']['singleSweep']['points']['points'] == [1, 2]
    assert jobs.create.call_args[1][
        'parent'] == 'projects/project-id/programs/test'
    assert jobs.get().execute.call_count == 1
    assert jobs.getResult().execute.call_count == 1


@mock.patch.object(discovery, 'build')
def test_bad_result_proto(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'
    }
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {
            'state': 'READY'
        }
    }
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {
            'state': 'SUCCESS'
        }
    }
    result = _RESULTS_V2.copy()
    result['@type'] = 'type.googleapis.com/unknown'
    jobs.getResult().execute.return_value = {'result': result}

    engine = cg.Engine(project_id='project-id',
                       proto_version=cg.engine.engine.ProtoVersion.V2)
    job = engine.run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'),
        params=cirq.Points('a', [1, 2]))
    with pytest.raises(ValueError, match='invalid result proto version'):
        job.results()


@mock.patch.object(discovery, 'build')
def test_bad_sweep_proto(build):
    eng = cg.Engine(project_id='project-id',
                    proto_version=cg.engine.engine.ProtoVersion.UNDEFINED)
    with pytest.raises(ValueError, match='invalid proto version'):
        eng.run(program=cirq.Circuit(),
                job_config=cg.JobConfig('project-id',
                                        gcs_prefix='gs://bucket/folder'))


@mock.patch.object(discovery, 'build')
def test_bad_priority(build):
    eng = cg.Engine(project_id='project-id',
                    proto_version=cg.engine.engine.ProtoVersion.V2)
    with pytest.raises(ValueError, match='priority must be'):
        eng.run(program=cirq.Circuit(),
                job_config=cg.JobConfig('project-id',
                                        gcs_prefix='gs://bucket/folder'),
                priority=1001)


@mock.patch.object(discovery, 'build')
def test_cancel(build):
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'}
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'READY'}}
    jobs.get().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {'state': 'CANCELLED'}}

    job = cg.Engine(project_id='project-id').run_sweep(
        program=cirq.Circuit(),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'))
    job.cancel()
    assert job.job_resource_name == ('projects/project-id/programs/test/'
                                     'jobs/test')
    assert job.status() == 'CANCELLED'
    assert jobs.cancel.call_args[1][
        'name'] == 'projects/project-id/programs/test/jobs/test'


@mock.patch.object(discovery, 'build')
def test_program_labels(build):
    program_name = 'projects/my-proj/programs/my-prog'
    service = mock.Mock()
    build.return_value = service
    programs = service.projects().programs()
    engine = cg.Engine(project_id='project-id')

    def body():
        return programs.patch.call_args[1]['body']

    programs.get().execute.return_value = {'labels': {'a': '1', 'b': '1'}}
    engine.add_program_labels(program_name, {'a': '2', 'c': '1'})

    assert body()['labels'] == {'a': '2', 'b': '1', 'c': '1'}
    assert body()['labelFingerprint'] == ''

    programs.get().execute.return_value = {'labels': {'a': '1', 'b': '1'},
                                           'labelFingerprint': 'abcdef'}
    engine.set_program_labels(program_name, {'s': '1', 'p': '1'})
    assert body()['labels'] == {'s': '1', 'p': '1'}
    assert body()['labelFingerprint'] == 'abcdef'

    programs.get().execute.return_value = {'labels': {'a': '1', 'b': '1'},
                                           'labelFingerprint': 'abcdef'}
    engine.remove_program_labels(program_name, ['a', 'c'])
    assert body()['labels'] == {'b': '1'}
    assert body()['labelFingerprint'] == 'abcdef'


@mock.patch.object(discovery, 'build')
def test_job_labels(build):
    job_name = 'projects/my-proj/programs/my-prog/jobs/my-job'
    service = mock.Mock()
    build.return_value = service
    jobs = service.projects().programs().jobs()
    engine = cg.Engine(project_id='project-id')

    def body():
        return jobs.patch.call_args[1]['body']

    jobs.get().execute.return_value = {'labels': {'a': '1', 'b': '1'}}
    engine.add_job_labels(job_name, {'a': '2', 'c': '1'})

    assert body()['labels'] == {'a': '2', 'b': '1', 'c': '1'}
    assert body()['labelFingerprint'] == ''

    jobs.get().execute.return_value = {'labels': {'a': '1', 'b': '1'},
                                       'labelFingerprint': 'abcdef'}
    engine.set_job_labels(job_name, {'s': '1', 'p': '1'})
    assert body()['labels'] == {'s': '1', 'p': '1'}
    assert body()['labelFingerprint'] == 'abcdef'

    jobs.get().execute.return_value = {'labels': {'a': '1', 'b': '1'},
                                       'labelFingerprint': 'abcdef'}
    engine.remove_job_labels(job_name, ['a', 'c'])
    assert body()['labels'] == {'b': '1'}
    assert body()['labelFingerprint'] == 'abcdef'


@mock.patch.object(discovery, 'build')
def test_implied_job_config_gcs_prefix(build):
    eng = cg.Engine(project_id='project_id')
    config = cg.JobConfig()

    # Implied by project id.
    assert eng.implied_job_config(config).gcs_prefix == 'gs://gqe-project_id/'

    # Bad default.
    eng_with_bad = cg.Engine(project_id='project-id',
                             default_gcs_prefix='bad_prefix')
    with pytest.raises(ValueError, match='gcs_prefix must be of the form'):
        _ = eng_with_bad.implied_job_config(config)

    # Good default without slash.
    eng_with = cg.Engine(project_id='project-id',
                         default_gcs_prefix='gs://good')
    assert eng_with.implied_job_config(config).gcs_prefix == 'gs://good/'

    # Good default with slash.
    eng_with = cg.Engine(project_id='project-id',
                         default_gcs_prefix='gs://good/')
    assert eng_with.implied_job_config(config).gcs_prefix == 'gs://good/'

    # Bad override.
    config.gcs_prefix = 'bad_prefix'
    with pytest.raises(ValueError, match='gcs_prefix must be of the form'):
        _ = eng.implied_job_config(config)
    with pytest.raises(ValueError, match='gcs_prefix must be of the form'):
        _ = eng_with_bad.implied_job_config(config)

    # Good override without slash.
    config.gcs_prefix = 'gs://better'
    assert eng.implied_job_config(config).gcs_prefix == 'gs://better/'
    assert eng_with.implied_job_config(config).gcs_prefix == 'gs://better/'

    # Good override with slash.
    config.gcs_prefix = 'gs://better/'
    assert eng.implied_job_config(config).gcs_prefix == 'gs://better/'
    assert eng_with.implied_job_config(config).gcs_prefix == 'gs://better/'


# uses re.fullmatch
@mock.patch.object(discovery, 'build')
def test_implied_job_config(build):
    eng = cg.Engine(project_id='project_id')

    # Infer all from project id.
    implied = eng.implied_job_config(cg.JobConfig())
    assert re.fullmatch(r'prog-[0-9A-Z]+', implied.program_id)
    assert implied.job_id == 'job-0'
    assert implied.gcs_prefix == 'gs://gqe-project_id/'
    assert re.fullmatch(
        r'gs://gqe-project_id/programs/prog-[0-9A-Z]+/prog-[0-9A-Z]+',
        implied.gcs_program)
    assert re.fullmatch(
        r'gs://gqe-project_id/programs/prog-[0-9A-Z]+/jobs/job-0',
        implied.gcs_results)

    # Force program id.
    implied = eng.implied_job_config(cg.JobConfig(program_id='j'))
    assert implied.program_id == 'j'
    assert implied.job_id == 'job-0'
    assert implied.gcs_prefix == 'gs://gqe-project_id/'
    assert implied.gcs_program == 'gs://gqe-project_id/programs/j/j'
    assert implied.gcs_results == 'gs://gqe-project_id/programs/j/jobs/job-0'

    # Force all.
    implied = eng.implied_job_config(
        cg.JobConfig(program_id='b',
                     job_id='c',
                     gcs_prefix='gs://d',
                     gcs_program='e',
                     gcs_results='f'))
    assert implied.program_id == 'b'
    assert implied.job_id == 'c'
    assert implied.gcs_prefix == 'gs://d/'
    assert implied.gcs_program == 'e'
    assert implied.gcs_results == 'f'


@mock.patch.object(discovery, 'build')
def test_bad_job_config_inference_order(build):
    eng = cg.Engine(project_id='project-id')
    config = cg.JobConfig()

    with pytest.raises(ValueError):
        eng._infer_gcs_results(config)
    with pytest.raises(ValueError):
        eng._infer_gcs_program(config)
    eng._infer_gcs_prefix(config)

    eng._infer_gcs_results(config)
    eng._infer_gcs_program(config)


@mock.patch.object(discovery, 'build')
def test_list_processors(build):
    service = mock.Mock()
    build.return_value = service
    PROCESSOR1 = {'name': 'projects/myproject/processors/xmonsim'},
    PROCESSOR2 = {'name': 'projects/myproject/processors/gmonsin'},
    processors = service.projects().processors()
    processors.list().execute.return_value = ({
        'processors': [PROCESSOR1, PROCESSOR2],
    })

    result = cg.Engine(project_id='myproject').list_processors()
    assert processors.list.call_args[1]['parent'] == 'projects/myproject'
    assert result == [PROCESSOR1, PROCESSOR2]


@mock.patch.object(discovery, 'build')
def test_latest_calibration(build):
    service = mock.Mock()
    build.return_value = service
    calibrations = service.projects().processors().calibrations()
    calibrations.list().execute.return_value = ({
        'calibrations': [_CALIBRATION]
    })
    calibration = cg.Engine(project_id='myproject').get_latest_calibration('x')
    assert calibrations.list.call_args[1][
        'parent'] == 'projects/myproject/processors/x'
    assert calibration.timestamp == 1562544000021
    assert set(calibration.get_metric_names()) == set(['xeb', 't1'])


@mock.patch.object(discovery, 'build')
def test_missing_latest_calibration(build):
    service = mock.Mock()
    build.return_value = service
    calibrations = service.projects().processors().calibrations()
    calibrations.list().execute.return_value = {}
    calibration = cg.Engine(project_id='myproject').get_latest_calibration('x')
    assert calibrations.list.call_args[1][
        'parent'] == 'projects/myproject/processors/x'
    assert not calibration


@mock.patch.object(discovery, 'build')
def test_calibration_from_job(build):
    service = mock.Mock()
    build.return_value = service

    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'
    }
    calibrationName = '/project/p/processor/x/calibrationsi/123'
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {
            'state': 'SUCCESS',
            'calibrationName': calibrationName,
        },
    }
    calibrations = service.projects().processors().calibrations()
    calibrations.get().execute.return_value = {'data': _CALIBRATION}

    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig(gcs_prefix='gs://bucket/folder'))

    calibration = job.get_calibration()
    assert calibration.timestamp == 1562544000021
    assert set(calibration.get_metric_names()) == set(['xeb', 't1'])
    assert calibrations.get.call_args[1]['name'] == calibrationName


@mock.patch.object(discovery, 'build')
def test_calibration_from_job_with_no_calibration(build):
    service = mock.Mock()
    build.return_value = service

    programs = service.projects().programs()
    jobs = programs.jobs()
    programs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test'
    }
    jobs.create().execute.return_value = {
        'name': 'projects/project-id/programs/test/jobs/test',
        'executionStatus': {
            'state': 'SUCCESS',
        },
    }

    calibrations = service.projects().processors().calibrations()
    engine = cg.Engine(project_id='project-id')
    job = engine.run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig(gcs_prefix='gs://bucket/folder'))

    calibration = job.get_calibration()
    assert not calibration
    assert not calibrations.get.called


@mock.patch.object(discovery, 'build')
def test_calibration_metrics(build):
    calibration = cg.engine.engine.Calibration(_CALIBRATION['data'])
    t1s = calibration.get_metrics_by_name('t1')
    xebs = calibration.get_metrics_by_name('xeb')

    assert t1s == [
        {
            'targets': ['q0_0'],
            'values': [321]
        },
        {
            'targets': ['q0_1'],
            'values': [911]
        },
        {
            'targets': ['q1_0'],
            'values': [505]
        },
    ]

    assert xebs == [
        {
            'targets': ['q0_0', 'q0_1'],
            'values': [.9999]
        },
        {
            'targets': ['q0_0', 'q1_0'],
            'values': [.9998]
        },
    ]


@mock.patch.object(discovery, 'build')
def test_alternative_api_and_key(build):
    disco = ('https://secret.googleapis.com/$discovery/rest?version=vfooo'
             '&key=my-key')
    cg.Engine(project_id='project-id', discovery_url=disco)
    build.assert_called_with(mock.ANY,
                             mock.ANY,
                             discoveryServiceUrl=disco,
                             requestBuilder=mock.ANY)


class MockRequestBuilder:

    def __init__(self):
        self.headers = {}


@mock.patch.object(discovery, 'build')
@mock.patch.object(http, 'HttpRequest')
def test_request_builder(HttpRequest, build):
    HttpRequest.return_value = MockRequestBuilder()
    cg.Engine(project_id='project-id')
    builtRequest = build.call_args[1]['requestBuilder']()
    assert builtRequest.headers['X-Goog-User-Project'] == 'project-id'


@mock.patch.object(discovery, 'build')
def test_not_both_version_and_discovery(build):
    with pytest.raises(ValueError, match='both specified'):
        cg.Engine(project_id='project-id',
                  version='vNo',
                  discovery_url='funkyDisco')

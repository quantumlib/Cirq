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

import numpy as np
import pytest

from apiclient import discovery

import cirq
import cirq.google as cg
from cirq.testing.mock import mock


_A_RESULT = {
    'sweepResults': [
        {
            'repetitions': 1,
            'measurementKeys': [
                {
                    'key': 'q',
                    'qubits': [{'row': 1, 'col': 1}]
                }
            ],
            'parameterizedResults': [
                {
                    'params': {'assignments': {'a': 1}},
                    'measurementResults': base64.b64encode(b'01')
                }
            ]
        }
    ]
}


_RESULTS = {
    'sweepResults': [
        {
            'repetitions': 1,
            'measurementKeys': [
                {
                    'key': 'q',
                    'qubits': [{'row': 1, 'col': 1}]
                }
            ],
            'parameterizedResults': [
                {
                    'params': {'assignments': {'a': 1}},
                    'measurementResults': base64.b64encode(b'01')
                },
                {
                    'params': {'assignments': {'a': 2}},
                    'measurementResults': base64.b64encode(b'01')
                }
            ]
        }
    ]
}


@cirq.testing.only_test_in_python3
def test_repr():
    v = cirq.google.JobConfig(project_id='my-project-id',
                              program_id='my-program-id',
                              job_id='my-job-id')

    assert repr(v) == ("JobConfig(project_id='my-project-id', "
                       "program_id='my-program-id', "
                       "job_id='my-job-id', gcs_prefix=None, "
                       "gcs_program=None, gcs_results=None)")


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

    result = cg.Engine(api_key="key").run(
        program=cirq.Circuit(),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'))
    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum', 'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}&key=key'))
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
        cg.Engine(api_key="key").run(program=circuit,
                                     job_config=cg.JobConfig('project-id'))


@mock.patch.object(discovery, 'build')
def test_schedule_device_validation_fails(build):
    scheduled_op = cirq.ScheduledOperation(time=None, duration=None,
                                           operation=cirq.H.on(
                                               cirq.NamedQubit("dorothy")))
    schedule = cirq.Schedule(device=cg.Foxtail,
                             scheduled_operations=[scheduled_op])

    with pytest.raises(ValueError):
        cg.Engine(api_key="key").run(
            program=schedule,
            job_config=cg.JobConfig('project-id'))


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
    result = cg.Engine(api_key="key").run(
        program=circuit,
        job_config=cg.JobConfig('project-id'))
    assert result.repetitions == 1


@mock.patch.object(discovery, 'build')
def test_unsupported_program_type(build):
    eng = cg.Engine(api_key="key")
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
        cg.Engine(api_key="key").run(
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

    result = cg.Engine(api_key="key").run(
        program=cirq.Circuit(),
        job_config=cg.JobConfig('org.com:project-id'))
    assert result.repetitions == 1
    assert result.params.param_dict == {'a': 1}
    assert result.measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum', 'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}&key=key'))
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

    job = cg.Engine(api_key="key").run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'),
        params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})])
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum', 'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}&key=key'))
    assert programs.create.call_args[1]['parent'] == 'projects/project-id'
    sweeps = programs.create.call_args[1]['body']['code']['parameter_sweeps']
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
def test_run_sweep_sweeps(build):
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

    job = cg.Engine(api_key="key").run_sweep(
        program=cirq.moment_by_moment_schedule(cirq.UnconstrainedDevice,
                                               cirq.Circuit()),
        job_config=cg.JobConfig('project-id', gcs_prefix='gs://bucket/folder'),
        params=cirq.Points('a', [1, 2]))
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    build.assert_called_with('quantum', 'v1alpha1',
                             discoveryServiceUrl=('https://{api}.googleapis.com'
                                                  '/$discovery/rest?version='
                                                  '{apiVersion}&key=key'))
    assert programs.create.call_args[1]['parent'] == 'projects/project-id'
    sweeps = programs.create.call_args[1]['body']['code']['parameter_sweeps']
    assert len(sweeps) == 1
    assert sweeps[0]['repetitions'] == 1
    assert sweeps[0]['sweep']['factors'][0]['sweeps'][0]['points'][
        'points'] == [1, 2]
    assert jobs.create.call_args[1][
        'parent'] == 'projects/project-id/programs/test'
    assert jobs.get().execute.call_count == 1
    assert jobs.getResult().execute.call_count == 1


@mock.patch.object(discovery, 'build')
def test_bad_priority(build):
    eng = cg.Engine(api_key="key")
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

    job = cg.Engine(api_key="key").run_sweep(
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
    engine = cg.Engine(api_key="key")

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
    engine = cg.Engine(api_key="key")

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
def test_implied_job_config_project_id(build):
    eng = cg.Engine(api_key="key")
    with pytest.raises(ValueError, match='project id'):
        _ = eng.implied_job_config(None)
    with pytest.raises(ValueError, match='project id'):
        _ = eng.implied_job_config(cg.JobConfig())
    assert eng.implied_job_config(
        cg.JobConfig(project_id='specific')).project_id == 'specific'

    eng_with = cg.Engine(api_key="key", default_project_id='default')

    # Fallback to default.
    assert eng_with.implied_job_config(None).project_id == 'default'

    # Override default.
    assert eng_with.implied_job_config(
        cg.JobConfig(project_id='specific')).project_id == 'specific'


@mock.patch.object(discovery, 'build')
def test_implied_job_config_gcs_prefix(build):
    eng = cg.Engine(api_key="key")
    config = cg.JobConfig(project_id='project_id')

    # Implied by project id.
    assert eng.implied_job_config(config).gcs_prefix == 'gs://gqe-project_id/'

    # Bad default.
    eng_with_bad = cg.Engine(api_key="key", default_gcs_prefix='bad_prefix')
    with pytest.raises(ValueError, match='gcs_prefix must be of the form'):
        _ = eng_with_bad.implied_job_config(config)

    # Good default without slash.
    eng_with = cg.Engine(api_key="key", default_gcs_prefix='gs://good')
    assert eng_with.implied_job_config(config).gcs_prefix == 'gs://good/'

    # Good default with slash.
    eng_with = cg.Engine(api_key="key", default_gcs_prefix='gs://good/')
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


@cirq.testing.only_test_in_python3  # uses re.fullmatch
@mock.patch.object(discovery, 'build')
def test_implied_job_config(build):
    eng = cg.Engine(api_key="key")

    # Infer all from project id.
    implied = eng.implied_job_config(cg.JobConfig(project_id='project_id'))
    assert implied.project_id == 'project_id'
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
    implied = eng.implied_job_config(cg.JobConfig(
        project_id='j',
        program_id='g'))
    assert implied.project_id == 'j'
    assert implied.program_id == 'g'
    assert implied.job_id == 'job-0'
    assert implied.gcs_prefix == 'gs://gqe-j/'
    assert implied.gcs_program == 'gs://gqe-j/programs/g/g'
    assert implied.gcs_results == 'gs://gqe-j/programs/g/jobs/job-0'

    # Force all.
    implied = eng.implied_job_config(cg.JobConfig(
        project_id='a',
        program_id='b',
        job_id='c',
        gcs_prefix='gs://d',
        gcs_program='e',
        gcs_results='f'))
    assert implied.project_id == 'a'
    assert implied.program_id == 'b'
    assert implied.job_id == 'c'
    assert implied.gcs_prefix == 'gs://d/'
    assert implied.gcs_program == 'e'
    assert implied.gcs_results == 'f'


@mock.patch.object(discovery, 'build')
def test_bad_job_config_inference_order(build):
    eng = cg.Engine(api_key="key")
    config = cg.JobConfig()

    with pytest.raises(ValueError):
        eng._infer_gcs_prefix(config)
    config.project_id = 'project'

    with pytest.raises(ValueError):
        eng._infer_gcs_results(config)
    with pytest.raises(ValueError):
        eng._infer_gcs_program(config)
    eng._infer_gcs_prefix(config)

    eng._infer_gcs_results(config)
    eng._infer_gcs_program(config)

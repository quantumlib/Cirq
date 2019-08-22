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

from apiclient import discovery

import cirq
import cirq.google as cg


def test_status():
    engine = mock.Mock()
    job = {'name': 'projects/a/programs/b/jobs/steve',
           'executionStatus': {'state': 'RUNNING'}}
    engine.get_job.return_value =  job
    job = cg.EngineJob(job_config=cg.JobConfig(job_id='steve'),
                       job=job,
                       engine=engine)
    assert job.status() == 'RUNNING'


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
        program=cirq.Circuit(),
        job_config=cg.JobConfig(gcs_prefix='gs://bucket/folder'))

    calibration = job.get_calibration()
    assert not calibration
    assert not calibrations.get.called

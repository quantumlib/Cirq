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

import cirq
import cirq.google as cg


def test_run_delegation():
    engine = mock.Mock()
    expected_result = mock.Mock()
    engine.create_job.return_value = [expected_result]

    program = cg.EngineProgram('projects/my-proj/programs/my-prog', engine)
    job_config = cg.JobConfig(job_id='steve')
    param_resolver = cirq.ParamResolver({})
    job = program.run(job_config=job_config,
                      repetitions=10,
                      param_resolver=param_resolver,
                      priority=100,
                      processor_ids=['mine'])
    assert job == expected_result


def test_run_sweeps_delegation():
    engine = mock.Mock()
    expected_job = mock.Mock()
    engine.create_job.return_value = expected_job

    program = cg.EngineProgram('projects/my-proj/programs/my-prog', engine)
    job_config = cg.JobConfig(job_id='steve')
    param_resolver = cirq.ParamResolver({})
    job = program.run_sweep(job_config=job_config,
                            repetitions=10,
                            params=param_resolver,
                            priority=100,
                            processor_ids=['mine'])
    assert job == expected_job


def test_str():
    program = cg.EngineProgram('projects/my-proj/programs/my-prog', mock.Mock())
    assert str(program) == 'EngineProgram(projects/my-proj/programs/my-prog)'

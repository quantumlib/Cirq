# Copyright 2021 The Cirq Developers
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
import cirq

from cirq_google.engine.simulated_local_program import SimulatedLocalProgram
from cirq_google.engine.simulated_local_job import SimulatedLocalJob


def test_id():
    program = SimulatedLocalProgram([cirq.Circuit()], None, program_id='program')
    assert program.id() == 'program'


def test_delete():
    program = SimulatedLocalProgram([cirq.Circuit()], None, program_id='program')
    job1 = SimulatedLocalJob(
        job_id='test_job1', processor_id='test1', parent_program=program, repetitions=100, sweeps=[]
    )
    job2 = SimulatedLocalJob(
        job_id='test_job2', processor_id='test1', parent_program=program, repetitions=100, sweeps=[]
    )
    job3 = SimulatedLocalJob(
        job_id='test_job3', processor_id='test1', parent_program=program, repetitions=100, sweeps=[]
    )
    program.add_job(job1.id(), job1)
    program.add_job(job2.id(), job2)
    program.add_job(job3.id(), job3)
    assert set(program.list_jobs()) == {job1, job2, job3}
    program.delete_job(job2.id())
    assert set(program.list_jobs()) == {job1, job3}
    program.delete(delete_jobs=True)
    assert program.list_jobs() == []

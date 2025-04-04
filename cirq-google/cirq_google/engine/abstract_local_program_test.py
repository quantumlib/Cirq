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
import datetime

import pytest

import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program import AbstractLocalProgram


class NothingProgram(AbstractLocalProgram):
    def delete(self, delete_jobs: bool = False) -> None:
        pass

    def delete_job(self, job_id: str) -> None:
        pass


def test_delete():
    program = NothingProgram([cirq.Circuit()], None)
    program.delete()


def test_init():
    with pytest.raises(ValueError, match='No circuits provided'):
        _ = NothingProgram([], None)


def test_jobs():
    program = NothingProgram([cirq.Circuit()], None)
    job1 = NothingJob(
        job_id='test', processor_id='test1', parent_program=program, repetitions=100, sweeps=[]
    )
    job2 = NothingJob(
        job_id='test', processor_id='test2', parent_program=program, repetitions=100, sweeps=[]
    )
    job3 = NothingJob(
        job_id='test', processor_id='test3', parent_program=program, repetitions=100, sweeps=[]
    )
    job2.set_labels({'color': 'blue', 'shape': 'square'})
    job3.set_labels({'color': 'green', 'shape': 'square'})

    # Use private variables for deterministic searches
    job1._create_time = datetime.datetime.fromtimestamp(1000)
    job2._create_time = datetime.datetime.fromtimestamp(2000)
    job3._create_time = datetime.datetime.fromtimestamp(3000)
    failure = quantum.ExecutionStatus.State.FAILURE
    success = quantum.ExecutionStatus.State.SUCCESS
    job1._status = failure
    job2._status = failure
    job3._status = success

    with pytest.raises(KeyError):
        program.get_job('jerb')
    program.add_job('jerb', job1)
    program.add_job('employ', job2)
    program.add_job('jobbies', job3)
    assert program.get_job('jerb') == job1
    assert program.get_job('employ') == job2
    assert program.get_job('jobbies') == job3

    assert set(program.list_jobs(has_labels={'shape': 'square'})) == {job2, job3}
    assert program.list_jobs(has_labels={'color': 'blue'}) == [job2]
    assert program.list_jobs(has_labels={'color': 'green'}) == [job3]
    assert program.list_jobs(has_labels={'color': 'yellow'}) == []

    assert set(program.list_jobs(created_before=datetime.datetime.fromtimestamp(3500))) == {
        job1,
        job2,
        job3,
    }
    assert set(program.list_jobs(created_before=datetime.datetime.fromtimestamp(2500))) == {
        job1,
        job2,
    }
    assert set(program.list_jobs(created_before=datetime.datetime.fromtimestamp(1500))) == {job1}
    assert program.list_jobs(created_before=datetime.datetime.fromtimestamp(500)) == []

    assert set(program.list_jobs(created_after=datetime.datetime.fromtimestamp(500))) == {
        job1,
        job2,
        job3,
    }
    assert set(program.list_jobs(created_after=datetime.datetime.fromtimestamp(1500))) == {
        job2,
        job3,
    }
    assert set(program.list_jobs(created_after=datetime.datetime.fromtimestamp(2500))) == {job3}
    assert program.list_jobs(created_after=datetime.datetime.fromtimestamp(3500)) == []

    assert set(program.list_jobs(execution_states={failure, success})) == {job1, job2, job3}
    assert program.list_jobs(execution_states={success}) == [job3]
    assert set(program.list_jobs(execution_states={failure})) == {job1, job2}
    ready = quantum.ExecutionStatus.State.READY
    assert program.list_jobs(execution_states={ready}) == []
    assert set(program.list_jobs(execution_states={})) == {job1, job2, job3}

    assert set(program.list_jobs(has_labels={'shape': 'square'}, execution_states={failure})) == {
        job2
    }


def test_create_update_time():
    program = NothingProgram([cirq.Circuit()], None)
    create_time = datetime.datetime.fromtimestamp(1000)
    update_time = datetime.datetime.fromtimestamp(2000)

    program._create_time = create_time
    program._update_time = update_time

    assert program.create_time() == create_time
    assert program.update_time() == update_time


def test_description_and_labels():
    program = NothingProgram([cirq.Circuit()], None)
    assert not program.description()
    program.set_description('nothing much')
    assert program.description() == 'nothing much'
    program.set_description('other desc')
    assert program.description() == 'other desc'
    assert program.labels() == {}
    program.set_labels({'key': 'green'})
    assert program.labels() == {'key': 'green'}
    program.add_labels({'door': 'blue', 'curtains': 'white'})
    assert program.labels() == {'key': 'green', 'door': 'blue', 'curtains': 'white'}
    program.remove_labels(['key', 'door'])
    assert program.labels() == {'curtains': 'white'}
    program.set_labels({'walls': 'gray'})
    assert program.labels() == {'walls': 'gray'}


def test_circuit():
    circuit1 = cirq.Circuit(cirq.X(cirq.LineQubit(1)))
    circuit2 = cirq.Circuit(cirq.Y(cirq.LineQubit(2)))
    program = NothingProgram([circuit1], None)
    assert program.batch_size() == 1
    assert program.get_circuit() == circuit1
    assert program.get_circuit(0) == circuit1
    assert program.batch_size() == 1
    program = NothingProgram([circuit1, circuit2], None)
    assert program.batch_size() == 2
    assert program.get_circuit(0) == circuit1
    assert program.get_circuit(1) == circuit2

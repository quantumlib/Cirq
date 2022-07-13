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
"""A helper for jobs that have been created on the Quantum Engine."""
from typing import Optional, Sequence, Tuple
import datetime
import cirq

from cirq_google.cloud import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.engine_result import EngineResult


class NothingJob(AbstractLocalJob):
    """Blank version of AbstractLocalJob for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = quantum.ExecutionStatus.State.READY

    def execution_status(self) -> quantum.ExecutionStatus.State:
        return self._status

    def failure(self) -> Optional[Tuple[str, str]]:
        return ('failed', 'failure code')  # coverage: ignore

    def cancel(self) -> None:
        pass

    def delete(self) -> None:
        pass

    async def batched_results_async(self) -> Sequence[Sequence[EngineResult]]:
        return []  # coverage: ignore

    async def results_async(self) -> Sequence[EngineResult]:
        return []  # coverage: ignore

    async def calibration_results_async(self) -> Sequence[CalibrationResult]:
        return []  # coverage: ignore


def test_description_and_labels():
    job = NothingJob(
        job_id='test', processor_id='pot_of_gold', parent_program=None, repetitions=100, sweeps=[]
    )
    assert job.id() == 'test'
    assert job.processor_ids() == ['pot_of_gold']
    assert not job.description()
    job.set_description('nothing much')
    assert job.description() == 'nothing much'
    job.set_description('other desc')
    assert job.description() == 'other desc'
    assert job.labels() == {}
    job.set_labels({'key': 'green'})
    assert job.labels() == {'key': 'green'}
    job.add_labels({'door': 'blue', 'curtains': 'white'})
    assert job.labels() == {'key': 'green', 'door': 'blue', 'curtains': 'white'}
    job.remove_labels(['key', 'door'])
    assert job.labels() == {'curtains': 'white'}
    job.set_labels({'walls': 'gray'})
    assert job.labels() == {'walls': 'gray'}


def test_reps_and_sweeps():
    job = NothingJob(
        job_id='test',
        processor_id='grill',
        parent_program=None,
        repetitions=100,
        sweeps=[cirq.Linspace('t', 0, 10, 0.1)],
    )
    assert job.get_repetitions_and_sweeps() == (100, [cirq.Linspace('t', 0, 10, 0.1)])


def test_create_update_time():
    job = NothingJob(
        job_id='test', processor_id='pot_of_gold', parent_program=None, repetitions=100, sweeps=[]
    )
    create_time = datetime.datetime.fromtimestamp(1000)
    update_time = datetime.datetime.fromtimestamp(2000)
    job._create_time = create_time
    job._update_time = update_time
    assert job.create_time() == create_time
    assert job.update_time() == update_time

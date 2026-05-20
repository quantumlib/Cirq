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

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import TYPE_CHECKING
from unittest import mock

import pytest

import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_job import AbstractLocalJob

if TYPE_CHECKING:
    from cirq_google.engine.engine_result import EngineResult


class NothingJob(AbstractLocalJob):
    """Blank version of AbstractLocalJob for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = quantum.ExecutionStatus.State.READY

    def execution_status(self) -> quantum.ExecutionStatus.State:
        return self._status

    def failure(self) -> tuple[str, str] | None:
        return ('failed', 'failure code')  # pragma: no cover

    def cancel(self) -> None:
        pass

    def delete(self) -> None:
        pass

    async def results_async(self) -> Sequence[EngineResult]:
        return []  # pragma: no cover


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
    # Single program (non-batch)
    mock_program = mock.Mock()
    mock_program.is_batch.return_value = False
    job = NothingJob(
        job_id='test',
        processor_id='grill',
        parent_program=mock_program,
        repetitions=100,
        sweeps=[cirq.Linspace('t', 0, 10, 0.1)],
    )
    assert job.get_repetitions_and_sweeps() == (100, [cirq.Linspace('t', 0, 10, 0.1)])
    assert job.get_repetitions_and_sweeps(0) == (100, [cirq.Linspace('t', 0, 10, 0.1)])
    with pytest.raises(IndexError, match="Job is not a batch job"):
        _ = job.get_repetitions_and_sweeps(1)

    # Batch program, shared sweep
    mock_program_batch = mock.Mock()
    mock_program_batch.is_batch.return_value = True
    mock_program_batch.batch_size.return_value = 2
    job_batch_shared = NothingJob(
        job_id='test',
        processor_id='grill',
        parent_program=mock_program_batch,
        repetitions=100,
        sweeps=[cirq.Linspace('t', 0, 10, 0.1)],
    )
    # Shared sweep, works with None or any index
    assert job_batch_shared.get_repetitions_and_sweeps() == (100, [cirq.Linspace('t', 0, 10, 0.1)])
    assert job_batch_shared.get_repetitions_and_sweeps(0) == (100, [cirq.Linspace('t', 0, 10, 0.1)])
    assert job_batch_shared.get_repetitions_and_sweeps(1) == (100, [cirq.Linspace('t', 0, 10, 0.1)])

    # Batch program, mapped sweeps
    job_batch_mapped = NothingJob(
        job_id='test',
        processor_id='grill',
        parent_program=mock_program_batch,
        repetitions=100,
        sweeps=[cirq.Linspace('t', 0, 10, 0.1), cirq.Linspace('u', 0, 5, 0.5)],
    )
    # Mapped sweeps, requires index
    with pytest.raises(ValueError, match="mapped sweeps"):
        _ = job_batch_mapped.get_repetitions_and_sweeps()
    assert job_batch_mapped.get_repetitions_and_sweeps(0) == (100, [cirq.Linspace('t', 0, 10, 0.1)])
    assert job_batch_mapped.get_repetitions_and_sweeps(1) == (100, [cirq.Linspace('u', 0, 5, 0.5)])
    with pytest.raises(IndexError, match="Index 2 out of range"):
        _ = job_batch_mapped.get_repetitions_and_sweeps(2)


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


def test_get_circuit():
    mock_program = mock.Mock()
    job = NothingJob(
        job_id='test',
        processor_id='pot_of_gold',
        parent_program=mock_program,
        repetitions=100,
        sweeps=[],
    )

    circuit = cirq.Circuit()
    mock_program.get_circuit.return_value = circuit
    assert job.get_circuit() == circuit
    mock_program.get_circuit.assert_called_with(None)
    assert job.get_circuit(1) == circuit
    mock_program.get_circuit.assert_called_with(1)

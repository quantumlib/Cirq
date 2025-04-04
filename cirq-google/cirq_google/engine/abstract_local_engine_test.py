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
from typing import Dict, List, Optional, Union

import pytest

import cirq
import cirq_google.engine.calibration as calibration
from cirq_google.engine.abstract_local_engine import AbstractLocalEngine
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_local_program_test import NothingProgram
from cirq_google.engine.abstract_program import AbstractProgram


class ProgramDictProcessor(AbstractLocalProcessor):
    """A processor that has a dictionary of programs for testing."""

    def __init__(self, programs: Dict[str, AbstractProgram], **kwargs):
        super().__init__(**kwargs)
        self._programs = programs

    def get_calibration(self, *args, **kwargs):
        pass

    def get_latest_calibration(self, timestamp: int) -> Optional[calibration.Calibration]:
        return calibration.Calibration()

    def get_current_calibration(self, *args, **kwargs):
        pass

    def get_device(self, *args, **kwargs):
        pass

    def get_device_specification(self, *args, **kwargs):
        pass

    def health(self, *args, **kwargs):
        pass

    def list_calibrations(self, *args, **kwargs):
        pass

    async def run_sweep_async(self, *args, **kwargs):
        pass

    def get_sampler(self, *args, **kwargs):
        return cirq.Simulator()

    def supported_languages(self, *args, **kwargs):
        pass

    def list_programs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ):
        """Lists all programs regardless of filters.

        This isn't really correct, but we don't want to test test functionality."""
        return self._programs.values()

    def get_program(self, program_id: str) -> AbstractProgram:
        return self._programs[program_id]


class NothingEngine(AbstractLocalEngine):
    """Engine for Testing."""

    def __init__(self, processors: List[AbstractLocalProcessor]):
        super().__init__(processors)


def test_get_processor():
    processor1 = ProgramDictProcessor(programs=[], processor_id='test')
    engine = NothingEngine([processor1])
    assert engine.get_processor('test') == processor1
    assert engine.get_processor('test').engine() == engine

    with pytest.raises(KeyError):
        _ = engine.get_processor('invalid')


def test_list_processor():
    processor1 = ProgramDictProcessor(programs=[], processor_id='proc')
    processor2 = ProgramDictProcessor(programs=[], processor_id='crop')
    engine = NothingEngine([processor1, processor2])
    assert engine.get_processor('proc') == processor1
    assert engine.get_processor('crop') == processor2
    assert engine.get_processor('proc').engine() == engine
    assert engine.get_processor('crop').engine() == engine
    assert set(engine.list_processors()) == {processor1, processor2}


def test_get_programs():
    program1 = NothingProgram([cirq.Circuit()], None)
    job1 = NothingJob(
        job_id='test3', processor_id='proc', parent_program=program1, repetitions=100, sweeps=[]
    )
    program1.add_job('jerb', job1)
    job1.add_labels({'color': 'blue'})

    program2 = NothingProgram([cirq.Circuit()], None)
    job2 = NothingJob(
        job_id='test4', processor_id='crop', parent_program=program2, repetitions=100, sweeps=[]
    )
    program2.add_job('jerb2', job2)
    job2.add_labels({'color': 'red'})

    processor1 = ProgramDictProcessor(programs={'prog1': program1}, processor_id='proc')
    processor2 = ProgramDictProcessor(programs={'prog2': program2}, processor_id='crop')
    engine = NothingEngine([processor1, processor2])

    assert engine.get_program('prog1') == program1

    with pytest.raises(KeyError, match='does not exist'):
        engine.get_program('invalid_id')

    assert set(engine.list_programs()) == {program1, program2}
    assert set(engine.list_jobs()) == {job1, job2}
    assert engine.list_jobs(has_labels={'color': 'blue'}) == [job1]
    assert engine.list_jobs(has_labels={'color': 'red'}) == [job2]

    program3 = NothingProgram([cirq.Circuit()], engine)
    assert program3.engine() == engine

    job3 = NothingJob(
        job_id='test5', processor_id='crop', parent_program=program3, repetitions=100, sweeps=[]
    )
    assert job3.program() == program3
    assert job3.engine() == engine
    assert job3.get_processor() == processor2
    assert job3.get_calibration() == calibration.Calibration()


def test_get_sampler():
    processor = ProgramDictProcessor(programs={}, processor_id='grocery')
    engine = NothingEngine([processor])
    assert isinstance(engine.get_sampler('grocery'), cirq.Sampler)
    with pytest.raises(ValueError, match='Invalid processor'):
        engine.get_sampler(['blah'])

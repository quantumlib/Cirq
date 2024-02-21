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
from typing import Dict, Optional, Union
import pytest

import cirq
import cirq_google
import sympy
import numpy as np

from cirq_google.api import v2
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program_test import NothingProgram
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor


class ProgramDictProcessor(AbstractLocalProcessor):
    """A processor that has a dictionary of programs for testing."""

    def __init__(self, programs: Dict[str, AbstractProgram], **kwargs):
        super().__init__(**kwargs)
        self._programs = programs

    def get_calibration(self, *args, **kwargs):
        pass

    def get_latest_calibration(self, *args, **kwargs):
        pass

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
        pass

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


def test_get_processor():
    processor1 = ProgramDictProcessor(programs=[], processor_id='test')
    engine = SimulatedLocalEngine([processor1])
    assert engine.get_processor('test') == processor1
    assert engine.get_processor('test').engine() == engine
    with pytest.raises(KeyError):
        engine.get_processor('abracadabra')
    with pytest.raises(ValueError, match='Invalid processor'):
        engine.get_sampler(processor_id=['a', 'b', 'c'])


def test_list_processor():
    processor1 = ProgramDictProcessor(programs=[], processor_id='proc')
    processor2 = ProgramDictProcessor(programs=[], processor_id='crop')
    engine = SimulatedLocalEngine([processor1, processor2])
    assert engine.get_processor('proc') == processor1
    assert engine.get_processor('crop') == processor2
    assert engine.get_processor('proc').engine() == engine
    assert engine.get_processor('crop').engine() == engine
    assert set(engine.list_processors()) == {processor1, processor2}


def test_get_programs():
    program1 = NothingProgram([cirq.Circuit()], None)
    job1 = NothingJob(
        job_id='test', processor_id='test1', parent_program=program1, repetitions=100, sweeps=[]
    )
    program1.add_job('jerb', job1)
    job1.add_labels({'color': 'blue'})

    program2 = NothingProgram([cirq.Circuit()], None)
    job2 = NothingJob(
        job_id='test', processor_id='test2', parent_program=program2, repetitions=100, sweeps=[]
    )
    program2.add_job('jerb2', job2)
    job2.add_labels({'color': 'red'})

    processor1 = ProgramDictProcessor(programs={'prog1': program1}, processor_id='proc')
    processor2 = ProgramDictProcessor(programs={'prog2': program2}, processor_id='crop')
    engine = SimulatedLocalEngine([processor1, processor2])

    assert engine.get_program('prog1') == program1

    with pytest.raises(KeyError, match='does not exis'):
        _ = engine.get_program('yoyo')

    assert set(engine.list_programs()) == {program1, program2}
    assert set(engine.list_jobs()) == {job1, job2}
    assert engine.list_jobs(has_labels={'color': 'blue'}) == [job1]
    assert engine.list_jobs(has_labels={'color': 'red'}) == [job2]


def test_full_simulation():
    engine = SimulatedLocalEngine([SimulatedLocalProcessor(processor_id='tester')])
    q = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    job = engine.get_processor('tester').run_sweep(circuit, params=sweep, repetitions=100)
    assert job.engine() == engine
    assert job.program().engine() == engine
    results = job.results()
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)


def test_sampler():
    engine = SimulatedLocalEngine([SimulatedLocalProcessor(processor_id='tester')])
    q = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    results = engine.get_sampler('tester').run_sweep(circuit, params=sweep, repetitions=100)
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)


def test_get_calibration_from_job():
    cal_proto = v2.metrics_pb2.MetricsSnapshot(timestamp_ms=10000)
    cal = cirq_google.Calibration(cal_proto)
    proc = SimulatedLocalProcessor(processor_id='test_proc', calibrations={10000: cal})
    engine = SimulatedLocalEngine([proc])
    job = engine.get_processor('test_proc').run_sweep(cirq.Circuit(), params={}, repetitions=100)
    assert job.get_processor() == proc
    assert job.get_calibration() == cal


def test_no_calibration_from_job():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    engine = SimulatedLocalEngine([proc])
    job = engine.get_processor('test_proc').run_sweep(cirq.Circuit(), params={}, repetitions=100)
    assert job.get_processor() == proc
    assert job.get_calibration() is None

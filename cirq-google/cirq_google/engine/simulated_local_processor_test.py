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
"""Tests for SimulatedLocalProcessor"""
from typing import List
import datetime
import pytest

import numpy as np
import sympy

import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor, VALID_LANGUAGES


def test_calibrations():
    now = datetime.datetime.now()
    future = int((datetime.datetime.now() + datetime.timedelta(hours=2)).timestamp())
    cal_proto1 = v2.metrics_pb2.MetricsSnapshot(timestamp_ms=10000)
    cal_proto2 = v2.metrics_pb2.MetricsSnapshot(timestamp_ms=20000)
    cal_proto3 = v2.metrics_pb2.MetricsSnapshot(timestamp_ms=future * 1000)
    cal1 = cirq_google.Calibration(cal_proto1)
    cal2 = cirq_google.Calibration(cal_proto2)
    cal3 = cirq_google.Calibration(cal_proto3)
    proc = SimulatedLocalProcessor(
        processor_id='test_proc', calibrations={10000: cal1, 20000: cal2, future: cal3}
    )
    assert proc.get_calibration(10000) == cal1
    assert proc.get_calibration(20000) == cal2
    assert proc.get_calibration(future) == cal3
    assert proc.get_current_calibration() == cal2
    assert proc.list_calibrations(earliest_timestamp=5000, latest_timestamp=15000) == [cal1]
    assert proc.list_calibrations(earliest_timestamp=15000, latest_timestamp=25000) == [cal2]
    assert proc.list_calibrations(
        earliest_timestamp=now, latest_timestamp=now + datetime.timedelta(hours=2)
    ) == [cal3]
    assert proc.list_calibrations(
        earliest_timestamp=datetime.date.today(), latest_timestamp=now + datetime.timedelta(hours=2)
    ) == [cal3]
    cal_list = proc.list_calibrations(latest_timestamp=25000)
    assert len(cal_list) == 2
    assert cal1 in cal_list
    assert cal2 in cal_list
    cal_list = proc.list_calibrations(earliest_timestamp=15000)
    assert len(cal_list) == 2
    assert cal2 in cal_list
    assert cal3 in cal_list
    cal_list = proc.list_calibrations()
    assert len(cal_list) == 3
    assert cal1 in cal_list
    assert cal2 in cal_list
    assert cal3 in cal_list


def test_accessors():
    proc = SimulatedLocalProcessor(processor_id='test_proc', device=cirq_google.Sycamore23)
    assert proc.health()
    assert proc.get_device() == cirq_google.Sycamore23
    assert proc.supported_languages() == VALID_LANGUAGES


def test_list_jobs():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    job1 = proc.run_sweep(cirq.Circuit(), params={}, repetitions=100)
    job2 = proc.run_sweep(cirq.Circuit(), params={}, repetitions=100)

    program1 = job1.program()
    program2 = job2.program()
    program1.set_labels({'color': 'green'})
    program2.set_labels({'color': 'red', 'shape': 'blue'})

    # Modify creation times in order to make search deterministic
    program1._create_time = datetime.datetime.fromtimestamp(1000)
    program2._create_time = datetime.datetime.fromtimestamp(2000)

    assert proc.list_programs(created_before=datetime.datetime.fromtimestamp(1500)) == [program1]
    assert proc.list_programs(created_after=datetime.datetime.fromtimestamp(1500)) == [program2]
    program_list = proc.list_programs(created_after=datetime.datetime.fromtimestamp(500))
    assert len(program_list) == 2
    assert program1 in program_list
    assert program2 in program_list
    assert proc.list_programs(has_labels={'color': 'yellow'}) == []
    assert proc.list_programs(has_labels={'color': 'green'}) == [program1]
    assert proc.list_programs(has_labels={'color': 'red'}) == [program2]
    assert proc.list_programs(has_labels={'shape': 'blue'}) == [program2]
    assert proc.list_programs(has_labels={'color': 'red', 'shape': 'blue'}) == [program2]


def test_delete():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    job1 = proc.run_sweep(cirq.Circuit(), params={}, repetitions=100)
    job2 = proc.run_sweep(cirq.Circuit(), params={}, repetitions=200)
    program1 = job1.program()
    program2 = job2.program()
    job1_id = job1.id()
    job2_id = job2.id()
    program1_id = program1.id()
    program2_id = program2.id()
    assert program1.get_job(job1_id) == job1
    assert program2.get_job(job2_id) == job2
    assert proc.get_program(program1_id) == program1
    assert proc.get_program(program2_id) == program2
    job1.delete()
    assert proc.get_program(program1_id) == program1
    with pytest.raises(KeyError, match='not found'):
        _ = program1.get_job(job1_id)
    program2.delete(delete_jobs=True)
    with pytest.raises(KeyError, match='not found'):
        _ = program2.get_job(job2_id)
    with pytest.raises(KeyError, match='not found'):
        _ = program2.get_job(program2_id)


def test_run():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    q = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q, key='m'))
    result = proc.run(circuit, repetitions=100)
    assert np.all(result.measurements['m'] == 1)


def test_run_sweep():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    q = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    job = proc.run_sweep(circuit, params=sweep, repetitions=100, program_id='abc', job_id='def')
    assert proc.get_program('abc') == job.program()
    assert proc.get_program('abc').get_job('def') == job
    assert job.execution_status() == quantum.ExecutionStatus.State.READY
    assert len(job) == 2
    assert np.all(job[0].measurements['m'] == 1)
    assert np.all(job[1].measurements['m'] == 0)

    # Test iteration
    for idx, result in enumerate(job):
        assert np.all(result.measurements['m'] == 1 - idx)

    assert job.execution_status() == quantum.ExecutionStatus.State.SUCCESS

    # with default program_id and job_id
    job = proc.run_sweep(circuit, params=sweep, repetitions=100)
    assert job.execution_status() == quantum.ExecutionStatus.State.READY
    results = job.results()
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)
    assert job.execution_status() == quantum.ExecutionStatus.State.SUCCESS


def test_run_batch():
    q = cirq.GridQubit(5, 4)
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    circuits = [
        cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m')),
        cirq.Circuit(cirq.X(q) ** sympy.Symbol('x'), cirq.measure(q, key='m2')),
    ]
    sweeps = [cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1])]
    job = proc.run_batch(circuits, params_list=sweeps, repetitions=100)
    assert job.execution_status() == quantum.ExecutionStatus.State.READY
    results = job.batched_results()
    assert np.all(results[0][0].measurements['m'] == 1)
    assert np.all(results[0][1].measurements['m'] == 0)
    assert np.all(results[1][0].measurements['m2'] == 0)
    assert np.all(results[1][1].measurements['m2'] == 1)
    assert job.execution_status() == quantum.ExecutionStatus.State.SUCCESS


def _no_y_gates(circuits: List[cirq.Circuit], sweeps: List[cirq.Sweepable], repetitions: int):
    for circuit in circuits:
        for moment in circuit:
            for op in moment:
                if op.gate == cirq.Y:
                    raise ValueError('No Y gates allowed!')


def test_device_validation():
    proc = SimulatedLocalProcessor(
        processor_id='test_proc', device=cirq_google.Sycamore23, validator=_no_y_gates
    )

    q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    job = proc.run_sweep(circuit, params=sweep, repetitions=100)
    with pytest.raises(ValueError, match='Qubit not on device'):
        job.results()
    # Test validation through sampler
    with pytest.raises(ValueError, match='Qubit not on device'):
        _ = proc.get_sampler().run_sweep(circuit, params=sweep, repetitions=100)


def test_additional_validation():
    proc = SimulatedLocalProcessor(
        processor_id='test_proc', device=cirq_google.Sycamore23, validator=_no_y_gates
    )
    q = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.Y(q), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    job = proc.run_sweep(circuit, params=sweep, repetitions=100)
    with pytest.raises(ValueError, match='No Y gates allowed!'):
        job.results()

    # Test validation through sampler
    with pytest.raises(ValueError, match='No Y gates allowed!'):
        _ = proc.get_sampler().run_sweep(circuit, params=sweep, repetitions=100)


def test_device_specification():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    assert proc.get_device_specification() is None
    device_spec = v2.device_pb2.DeviceSpecification()
    device_spec.valid_qubits.append('q0_0')
    device_spec.valid_qubits.append('q0_1')
    proc = SimulatedLocalProcessor(processor_id='test_proc', device_specification=device_spec)
    assert proc.get_device_specification() == device_spec


def test_unsupported():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    with pytest.raises(NotImplementedError):
        _ = proc.run_calibration()

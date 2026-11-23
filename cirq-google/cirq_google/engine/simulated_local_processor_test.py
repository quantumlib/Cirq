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

from __future__ import annotations

import datetime

import numpy as np
import pytest
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
    result = proc.run(circuit, repetitions=100, run_name="run", device_config_name="config_alias")
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


def _no_y_gates(circuits: list[cirq.Circuit], sweeps: list[cirq.Sweepable], repetitions: int):
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


def test_simulated_local_processor_run_sweep_multi():
    processor = SimulatedLocalProcessor(processor_id='test_proc')
    circuit = cirq.Circuit(cirq.measure(cirq.GridQubit(0, 0), key='m'))

    # Mapping
    processor.run_sweep({'a': circuit}, params={}, repetitions=1)

    # Sequence (else block)
    processor.run_sweep([circuit], params={}, repetitions=1)


def test_simulated_local_processor_mapped_sweeps():
    Q = cirq.GridQubit(2, 2)
    # Circuit 0 uses symbol 't', Circuit 1 uses symbol 'u'
    circuit0 = cirq.Circuit(cirq.X(Q) ** sympy.Symbol('t'), cirq.measure(Q, key='m'))
    circuit1 = cirq.Circuit(cirq.Y(Q) ** sympy.Symbol('u'), cirq.measure(Q, key='m'))

    processor = SimulatedLocalProcessor(processor_id='test_processor')

    sweeps = [cirq.Points(key='t', points=[0, 1]), cirq.Points(key='u', points=[0, 1])]

    # We want to run:
    # - circuit0 with sweep over 't' (10 reps)
    # - circuit1 with sweep over 'u' (20 reps)

    job = processor.run_sweep(
        program=[circuit0, circuit1],
        params=sweeps,
        repetitions=[10, 20],
        device_config_name='test_config',
    )

    results = job.results()
    assert len(results) == 4  # 2 circuits * 2 sweep points each = 4 results

    # Engine returns results grouped by program then by sweep.
    # e.g. (P0, S0_0), (P0, S0_1), (P1, S1_0), (P1, S1_1)

    # P0 (circuit0, has 't') was run with 10 reps.
    assert len(results[0].measurements['m']) == 10
    assert len(results[1].measurements['m']) == 10

    # P1 (circuit1, has 'u') was run with 20 reps.
    assert len(results[2].measurements['m']) == 20
    assert len(results[3].measurements['m']) == 20

    # Verify values to ensure correct sweeps were mapped
    # P0 S0_0: t=0 -> X^0 = I -> measurements should be 0
    # P0 S0_1: t=1 -> X^1 = X -> measurements should be 1
    assert np.all(results[0].measurements['m'] == 0)
    assert np.all(results[1].measurements['m'] == 1)

    # P1 S1_0: u=0 -> Y^0 = I -> measurements should be 0
    # P1 S1_1: u=1 -> Y^1 = Y -> measurements should be 1
    assert np.all(results[2].measurements['m'] == 0)
    assert np.all(results[3].measurements['m'] == 1)


def test_simulated_local_processor_multiple_sweeps_single_circuit():
    Q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(Q) ** sympy.Symbol('t'), cirq.measure(Q, key='m'))

    processor = SimulatedLocalProcessor(processor_id='test_processor')

    sweeps = [cirq.Points(key='t', points=[0, 1]), cirq.Points(key='t', points=[1, 0])]

    job = processor.run_sweep(
        program=circuit, params=sweeps, repetitions=10, device_config_name='test_config'
    )

    results = job.results()
    assert len(results) == 4

    reps, job_sweeps = job.get_repetitions_and_sweeps()
    assert reps == 10
    assert len(job_sweeps) == 2
    assert job_sweeps[0] == sweeps[0]
    assert job_sweeps[1] == sweeps[1]

    assert np.all(results[0].measurements['m'] == 0)
    assert np.all(results[1].measurements['m'] == 1)
    assert np.all(results[2].measurements['m'] == 1)
    assert np.all(results[3].measurements['m'] == 0)

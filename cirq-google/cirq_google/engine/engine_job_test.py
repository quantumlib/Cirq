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

from __future__ import annotations

import datetime
from http import HTTPStatus
from unittest import mock

import duet
import pytest
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge

import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.cloud import quantum
from cirq_google.engine import EngineException, util
from cirq_google.engine.engine import EngineContext
from cirq_google.engine.stream_manager import StreamError

_PROGRAM_V2 = util.pack_any(
    Merge(
        """language {
  gate_set: "v2_5"
  arg_function_language: "exp"
}
circuit {
  scheduling_strategy: MOMENT_BY_MOMENT
  moments {
    operations {
      qubit_constant_index: 0
      phasedxpowgate {
        phase_exponent {
          float_value: 0.0
        }
        exponent {
          float_value: 0.5
        }
      }
    }
  }
  moments {
    operations {
      qubit_constant_index: 0
      measurementgate {
        key {
          arg_value {
            string_value: "result"
          }
        }
        invert_mask {
          arg_value {
            bool_values {
            }
          }
        }
      }
    }
  }
}
constants {
  qubit {
    id: "5_2"
  }
}
""",
        v2.program_pb2.Program(),
    )
)

_BATCH_PROGRAM_V2 = util.pack_any(
    Merge(
        """language {
  gate_set: "v2_5"
  arg_function_language: "exp"
}
keyed_circuits {
  key: "c1"
  circuit {
    scheduling_strategy: MOMENT_BY_MOMENT
    moments {
      operations {
        qubit_constant_index: 0
        phasedxpowgate {
          phase_exponent {
            float_value: 0.0
          }
          exponent {
            float_value: 0.5
          }
        }
      }
    }
  }
}
keyed_circuits {
  key: "c2"
  circuit {
    scheduling_strategy: MOMENT_BY_MOMENT
    moments {
      operations {
        qubit_constant_index: 0
        phasedxpowgate {
          phase_exponent {
            float_value: 0.0
          }
          exponent {
            float_value: 0.5
          }
        }
      }
    }
  }
}
constants {
  qubit {
    id: "5_2"
  }
}
""",
        v2.program_pb2.Program(),
    )
)


@pytest.fixture(scope='module', autouse=True)
def mock_grpc_client():
    with mock.patch(
        'cirq_google.engine.engine_client.quantum.QuantumEngineServiceClient'
    ) as _fixture:
        yield _fixture


def test_engine():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.engine().project_id == 'a'


def test_program():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.program().project_id == 'a'
    assert job.program().program_id == 'b'


def test_id():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(create_time=timestamp_pb2.Timestamp(seconds=1581515101)),
    )
    assert job.id() == 'steve'


def test_create_time():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(create_time=timestamp_pb2.Timestamp(seconds=1581515101)),
    )
    assert job.create_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_update_time(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(
        update_time=timestamp_pb2.Timestamp(seconds=1581515101)
    )
    assert job.update_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )
    get_job.assert_called_once_with('a', 'b', 'steve', False)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_description(get_job):
    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(description='hello')
    )
    assert job.description() == 'hello'
    get_job.return_value = quantum.QuantumJob(description='hello')
    assert cg.EngineJob('a', 'b', 'steve', EngineContext()).description() == 'hello'
    get_job.assert_called_once_with('a', 'b', 'steve', False)


@mock.patch('cirq_google.engine.engine_client.EngineClient.set_job_description_async')
def test_set_description(set_job_description):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    set_job_description.return_value = quantum.QuantumJob(description='world')
    assert job.set_description('world').description() == 'world'
    set_job_description.assert_called_with('a', 'b', 'steve', 'world')

    set_job_description.return_value = quantum.QuantumJob(description='')
    assert job.set_description('').description() == ''
    set_job_description.assert_called_with('a', 'b', 'steve', '')


def test_labels():
    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(labels={'t': '1'})
    )
    assert job.labels() == {'t': '1'}


@mock.patch('cirq_google.engine.engine_client.EngineClient.set_job_labels_async')
def test_set_labels(set_job_labels):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    set_job_labels.return_value = quantum.QuantumJob(labels={'a': '1', 'b': '1'})
    assert job.set_labels({'a': '1', 'b': '1'}).labels() == {'a': '1', 'b': '1'}
    set_job_labels.assert_called_with('a', 'b', 'steve', {'a': '1', 'b': '1'})

    set_job_labels.return_value = quantum.QuantumJob()
    assert job.set_labels({}).labels() == {}
    set_job_labels.assert_called_with('a', 'b', 'steve', {})


@mock.patch('cirq_google.engine.engine_client.EngineClient.add_job_labels_async')
def test_add_labels(add_job_labels):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(labels={}))
    assert job.labels() == {}

    add_job_labels.return_value = quantum.QuantumJob(labels={'a': '1'})
    assert job.add_labels({'a': '1'}).labels() == {'a': '1'}
    add_job_labels.assert_called_with('a', 'b', 'steve', {'a': '1'})

    add_job_labels.return_value = quantum.QuantumJob(labels={'a': '2', 'b': '1'})
    assert job.add_labels({'a': '2', 'b': '1'}).labels() == {'a': '2', 'b': '1'}
    add_job_labels.assert_called_with('a', 'b', 'steve', {'a': '2', 'b': '1'})


@mock.patch('cirq_google.engine.engine_client.EngineClient.remove_job_labels_async')
def test_remove_labels(remove_job_labels):
    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(labels={'a': '1', 'b': '1'})
    )
    assert job.labels() == {'a': '1', 'b': '1'}

    remove_job_labels.return_value = quantum.QuantumJob(labels={'b': '1'})
    assert job.remove_labels(['a']).labels() == {'b': '1'}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a'])

    remove_job_labels.return_value = quantum.QuantumJob(labels={})
    assert job.remove_labels(['a', 'b', 'c']).labels() == {}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a', 'b', 'c'])


def test_processor_ids():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(
            scheduling_config=quantum.SchedulingConfig(
                processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                    processor_names=['projects/a/processors/p']
                )
            )
        ),
    )
    assert job.processor_ids() == ['p']


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_status(get_job):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.RUNNING)
    )
    get_job.return_value = qjob

    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.status() == 'RUNNING'
    get_job.assert_called_once()
    assert job.execution_status() == quantum.ExecutionStatus.State.RUNNING


def test_failure():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(
            execution_status=quantum.ExecutionStatus(
                state=quantum.ExecutionStatus.State.FAILURE,
                failure=quantum.ExecutionStatus.Failure(
                    error_code=quantum.ExecutionStatus.Failure.Code.SYSTEM_ERROR,
                    error_message='boom',
                ),
            )
        ),
    )
    assert job.failure() == ('SYSTEM_ERROR', 'boom')


def test_failure_with_no_error():
    job = cg.EngineJob(
        'a',
        'b',
        'steve',
        EngineContext(),
        _job=quantum.QuantumJob(
            execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS)
        ),
    )
    assert not job.failure()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps(get_job, get_program):
    # Single program (non-batch)
    get_program.return_value = quantum.QuantumProgram(code=_PROGRAM_V2)
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=10)]
            )
        )
    )
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job.get_repetitions_and_sweeps() == (10, [cirq.UnitSweep])
    assert job.get_repetitions_and_sweeps(0) == (10, [cirq.UnitSweep])
    with pytest.raises(IndexError, match="Job is not a batch job"):
        _ = job.get_repetitions_and_sweeps(1)

    # Batch program, shared sweep
    get_program.return_value = quantum.QuantumProgram(code=_BATCH_PROGRAM_V2)
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=10)]
            )
        )
    )
    job_batch_shared = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert job_batch_shared.get_repetitions_and_sweeps() == (10, [cirq.UnitSweep])
    assert job_batch_shared.get_repetitions_and_sweeps(0) == (10, [cirq.UnitSweep])
    assert job_batch_shared.get_repetitions_and_sweeps(1) == (10, [cirq.UnitSweep])

    # Batch program, mapped sweeps
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[
                    v2.run_context_pb2.ParameterSweep(repetitions=10),
                    v2.run_context_pb2.ParameterSweep(
                        repetitions=10,
                        sweep=v2.run_context_pb2.Sweep(
                            single_sweep=v2.run_context_pb2.SingleSweep(
                                parameter_key='t',
                                points=v2.run_context_pb2.Points(points_double=[1.0, 2.0]),
                            )
                        ),
                    ),
                ]
            )
        )
    )
    job_batch_mapped = cg.EngineJob('a', 'b', 'steve', EngineContext())
    # Mapped sweeps, requires index
    with pytest.raises(ValueError, match="mapped sweeps"):
        _ = job_batch_mapped.get_repetitions_and_sweeps()
    assert job_batch_mapped.get_repetitions_and_sweeps(0) == (10, [cirq.UnitSweep])
    assert job_batch_mapped.get_repetitions_and_sweeps(1) == (10, [cirq.Points('t', [1.0, 2.0])])
    with pytest.raises(IndexError, match="Index 2 out of range"):
        _ = job_batch_mapped.get_repetitions_and_sweeps(2)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps_non_uniform(get_job, get_program):
    get_program.return_value = quantum.QuantumProgram(code=_BATCH_PROGRAM_V2)
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(
            v2.run_context_pb2.RunContext(
                parameter_sweeps=[
                    v2.run_context_pb2.ParameterSweep(repetitions=10),
                    v2.run_context_pb2.ParameterSweep(
                        repetitions=20,
                        sweep=v2.run_context_pb2.Sweep(
                            single_sweep=v2.run_context_pb2.SingleSweep(
                                parameter_key='t',
                                points=v2.run_context_pb2.Points(points_double=[1.0, 2.0]),
                            )
                        ),
                    ),
                ]
            )
        )
    )
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())

    with pytest.raises(ValueError, match="mapped sweeps"):
        _ = job.get_repetitions_and_sweeps()

    reps0, sweeps0 = job.get_repetitions_and_sweeps(0)
    assert reps0 == 10
    assert sweeps0 == [cirq.UnitSweep]

    reps1, sweeps1 = job.get_repetitions_and_sweeps(1)
    assert reps1 == 20
    assert sweeps1 == [cirq.Points('t', [1.0, 2.0])]


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps_v1(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(
            v1.program_pb2.RunContext(
                parameter_sweeps=[v1.params_pb2.ParameterSweep(repetitions=10)]
            )
        )
    )
    with pytest.raises(ValueError, match='v1 RunContext is not supported'):
        job.get_repetitions_and_sweeps()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps_unsupported(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(
        run_context=any_pb2.Any(type_url='type.googleapis.com/unknown.proto')
    )
    with pytest.raises(ValueError, match='unsupported run_context type: unknown.proto'):
        job.get_repetitions_and_sweeps()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps_no_repetitions(get_job, get_program):
    get_program.return_value = quantum.QuantumProgram(code=_PROGRAM_V2)
    get_job.return_value = quantum.QuantumJob(
        run_context=util.pack_any(v2.run_context_pb2.RunContext(parameter_sweeps=[]))
    )
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    with pytest.raises(ValueError, match="No repetitions found in run context."):
        _ = job.get_repetitions_and_sweeps()


def test_get_processor():
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(processor_name='projects/a/processors/p')
    )

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert job.get_processor().processor_id == 'p'


def test_get_processor_no_processor():
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus())

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert not job.get_processor()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_calibration_async')
def test_get_calibration(get_calibration, get_job):
    calibration_timestamp_seconds = 1562
    raw_calibration_data = """
timestamp_ms: 1562,
metrics: [{
    name: 'two_qubit_xeb',
    targets: ['0_0', '0_1'],
    values: [{
        double_val: .9999
    }]
    }, {
    name: 'two_qubit_xeb',
    targets: ['0_0', '1_0'],
    values: [{
        double_val: .9998
    }]
    }, {
    name: 't1',
    targets: ['0_0'],
    values: [{
        double_val: 321
    }]
    }, {
    name: 't1',
    targets: ['0_1'],
    values: [{
        double_val: 911
    }]
    }, {
    name: 't1',
    targets: ['1_0'],
    values: [{
        double_val: 505
    }]
    }, {
    name: 'globalMetric',
    values: [{
        int32_val: 12300
    }]
}]
"""
    calibration = Merge(raw_calibration_data, v2.metrics_pb2.MetricsSnapshot())
    job_calibration = cg.Calibration(calibration=calibration)

    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(
            state=quantum.ExecutionStatus.State.SUCCESS,
            calibration_name=f"projects/a/processors/p/calibrations/{calibration_timestamp_seconds}",
        )
    )
    get_job.return_value = qjob

    calibration_any_proto = any_pb2.Any()
    calibration_any_proto.Pack(calibration)
    quantum_calibration = quantum.QuantumCalibration(
        name='calibration1',
        timestamp=timestamp_pb2.Timestamp(seconds=calibration_timestamp_seconds),
        data=calibration_any_proto,
    )

    get_calibration.return_value = quantum_calibration

    job = cg.EngineJob('a', 'b', 'steve', EngineContext())

    assert job.get_calibration() == job_calibration
    get_job.assert_called_once()
    get_calibration.assert_called_once_with('a', 'p', calibration_timestamp_seconds)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_calibration_no_calibration(get_job):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.RUNNING)
    )
    get_job.return_value = qjob

    job = cg.EngineJob('a', 'b', 'steve', EngineContext())

    assert not job.get_calibration()
    get_job.assert_called_once()


def test_get_config_not_set():
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS)
    )
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    with pytest.raises(ValueError, match="device_config_key is not set.*state: SUCCESS"):
        _ = job.get_config()


def test_get_config_non_terminal():
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.RUNNING)
    )
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    with pytest.warns(UserWarning, match="RUNNING"):
        assert job.get_config() is None


@pytest.mark.parametrize(
    "key_kwargs", [{"config_alias": "alias1"}, {"run": "run1"}, {"snapshot_id": "snapshot1"}]
)
def test_get_config_incomplete(key_kwargs):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(
            state=quantum.ExecutionStatus.State.SUCCESS,
            device_config_key=quantum.DeviceConfigKey(**key_kwargs),
        )
    )
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    with pytest.raises(
        ValueError, match="must have both `config_alias` and either `snapshot_id` or `run` set"
    ):
        _ = job.get_config()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_quantum_processor_config_async')
def test_get_config_success(get_quantum_processor_config, get_job):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(
            state=quantum.ExecutionStatus.State.SUCCESS,
            processor_name='projects/a/processors/p',
            device_config_key=quantum.DeviceConfigKey(run='run1', config_alias='alias1'),
        )
    )

    get_job.return_value = qjob

    mock_config = quantum.QuantumProcessorConfig(
        name='projects/a/processors/p/configSnapshots/snapshot1/configs/alias1'
    )
    get_quantum_processor_config.return_value = mock_config

    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    config = job.get_config()

    assert config is not None
    assert config._quantum_processor_config == mock_config
    get_job.assert_called_once()
    get_quantum_processor_config.assert_called_once_with(
        project_id='a',
        processor_id='p',
        device_config_revision=cg.Run('run1'),
        config_name='alias1',
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_quantum_processor_config_async')
def test_get_config_snapshot_success(get_quantum_processor_config, get_job):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(
            state=quantum.ExecutionStatus.State.SUCCESS,
            processor_name='projects/a/processors/p',
            device_config_key=quantum.DeviceConfigKey(
                snapshot_id='snapshot1', config_alias='alias1'
            ),
        )
    )
    get_job.return_value = qjob

    mock_config = quantum.QuantumProcessorConfig(
        name='projects/a/processors/p/configSnapshots/snapshot1/configs/alias1'
    )
    get_quantum_processor_config.return_value = mock_config

    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    config = job.get_config()

    assert config is not None
    assert config._quantum_processor_config == mock_config
    get_job.assert_called_once()
    get_quantum_processor_config.assert_called_once_with(
        project_id='a',
        processor_id='p',
        device_config_revision=cg.Snapshot('snapshot1'),
        config_name='alias1',
    )


def test_get_config_no_processor_name():
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(
            state=quantum.ExecutionStatus.State.SUCCESS,
            device_config_key=quantum.DeviceConfigKey(
                snapshot_id='snapshot1', config_alias='alias1'
            ),
        )
    )
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    with pytest.raises(ValueError, match="Processor name is not set in job status."):
        _ = job.get_config()


@mock.patch('cirq_google.engine.engine_client.EngineClient.cancel_job_async')
def test_cancel(cancel_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    job.cancel()
    cancel_job.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_job_async')
def test_delete(delete_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    job.delete()
    delete_job.assert_called_once_with('a', 'b', 'steve')


RESULTS = quantum.QuantumResult(
    result=util.pack_any(
        Merge(
            """
sweep_results: [{
        repetitions: 4,
        parameterized_results: [{
            params: {
                assignments: {
                    key: 'a'
                    value: 1
                }
            },
            measurement_results: {
                key: 'q'
                qubit_measurement_results: [{
                  qubit: {
                    id: '1_1'
                  }
                  results: '\006'
                }]
            }
        },{
            params: {
                assignments: {
                    key: 'a'
                    value: 2
                }
            },
            measurement_results: {
                key: 'q'
                qubit_measurement_results: [{
                  qubit: {
                    id: '1_1'
                  }
                  results: '\005'
                }]
            }
        }]
    }]
""",
            v2.result_pb2.Result(),
        )
    )
)

RESULTS_NON_UNIFORM = quantum.QuantumResult(
    result=util.pack_any(
        Merge(
            """
sweep_results: [{
        repetitions: 10,
        parameterized_results: [{
            params: {
                assignments: {
                    key: 'a'
                    value: 1
                }
            },
            measurement_results: {
                key: 'q'
                qubit_measurement_results: [{
                  qubit: {
                    id: '1_1'
                  }
                  results: "\\000\\000"
                }]
            }
        }]
    }, {
        repetitions: 20,
        parameterized_results: [{
            params: {
                assignments: {
                    key: 'a'
                    value: 2
                }
            },
            measurement_results: {
                key: 'q'
                qubit_measurement_results: [{
                  qubit: {
                    id: '1_1'
                  }
                  results: "\\000\\000\\000"
                }]
            }
        }]
    }]
""",
            v2.result_pb2.Result(),
        )
    )
)


UPDATE_TIME = datetime.datetime.now(tz=datetime.timezone.utc)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    data = job.results()
    assert len(data) == 2
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'
    get_job_results.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_non_uniform(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS_NON_UNIFORM

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    data = job.results()
    assert len(data) == 2
    assert len(data[0].measurements['q']) == 10
    assert len(data[1].measurements['q']) == 20
    # Verify they are correct EngineResult
    assert data[0].job_id == 'steve'
    assert data[1].job_id == 'steve'


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_iter(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    results = [str(r) for r in job]
    assert len(results) == 2
    assert results[0] == 'q=0110'
    assert results[1] == 'q=1010'


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_getitem(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert str(job[0]) == 'q=0110'
    assert str(job[1]) == 'q=1010'
    with pytest.raises(IndexError):
        _ = job[2]


def test_receives_results_via_stream_returns_correct_results():
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    result_future = duet.completed_future(RESULTS)

    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=qjob, job_result_future=result_future
    )
    data = job.results()

    assert len(data) == 2
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'


def test_receives_job_via_stream_raises_and_updates_underlying_job():
    expected_error_code = quantum.ExecutionStatus.Failure.Code.SYSTEM_ERROR
    expected_error_message = 'system error'
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(
            state=quantum.ExecutionStatus.State.SUCCESS,
            failure=quantum.ExecutionStatus.Failure(
                error_code=expected_error_code, error_message=expected_error_message
            ),
        ),
        update_time=UPDATE_TIME,
    )
    result_future = duet.completed_future(qjob)

    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=qjob, job_result_future=result_future
    )
    qjob.execution_status.state = quantum.ExecutionStatus.State.FAILURE

    with pytest.raises(RuntimeError):
        job.results()
    actual_error_code, actual_error_message = job.failure()

    # Checks that the underlying job has been updated by checking failure information.
    assert actual_error_code == expected_error_code.name
    assert actual_error_message == expected_error_message


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_on_stream_failure_retrieves_results_using_get_job_results(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS
    result_future = duet.failed_future(StreamError(RuntimeError("stream failed")))

    job = cg.EngineJob(
        'a', 'b', 'steve', EngineContext(), _job=qjob, job_result_future=result_future
    )
    data = job.results()

    assert len(data) == 2
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'
    get_job_results.assert_called_once_with('a', 'b', 'steve')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_recreate_job_if_not_found(get_job_results, get_job):
    project_id = 'a'
    program_id = 'b'
    job_id = 'steve'
    context = EngineContext(timeout=60, enable_streaming=False)

    get_job.side_effect = EngineException('job not found', HTTPStatus.NOT_FOUND)

    async def recreate_job():
        qjob = quantum.QuantumJob(
            execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
            update_time=UPDATE_TIME,
        )
        get_job.side_effect = None
        get_job.return_value = qjob
        get_job_results.return_value = RESULTS
        return cg.EngineJob(
            project_id=project_id,
            program_id=program_id,
            job_id=job_id,
            context=context,
            _job=qjob,
            job_result_future=None,
            recreate_job=None,
        )

    job = cg.EngineJob(
        project_id=project_id,
        program_id=program_id,
        job_id=job_id,
        context=context,
        _job=None,
        job_result_future=None,
        recreate_job=recreate_job,
    )
    data = job.results()

    assert len(data) == 2
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'
    get_job.assert_has_calls((mock.call(project_id, program_id, job_id, False),))
    get_job_results.assert_called_once_with(project_id, program_id, job_id)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_receive_results_get_job_error_propagated(get_job_results, get_job):
    project_id = 'a'
    program_id = 'b'
    job_id = 'steve'
    context = EngineContext(timeout=60, enable_streaming=False)

    get_job.side_effect = EngineException('internal error', HTTPStatus.INTERNAL_SERVER_ERROR)

    job = cg.EngineJob(
        project_id=project_id,
        program_id=program_id,
        job_id=job_id,
        context=context,
        _job=None,
        job_result_future=None,
    )

    with pytest.raises(EngineException) as exc_info:
        job.results()
    assert exc_info.value.code == HTTPStatus.INTERNAL_SERVER_ERROR


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_len(get_job_results):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS

    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert len(job) == 2


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_timeout(get_job):
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.RUNNING),
        update_time=UPDATE_TIME,
    )
    get_job.return_value = qjob
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(timeout=0.1))
    with pytest.raises(TimeoutError):
        job.results()


def test_str():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    assert str(job) == 'EngineJob(project_id=\'a\', program_id=\'b\', job_id=\'steve\')'


def test_get_circuit():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    circuit = cirq.Circuit()
    with mock.patch.object(
        cg.EngineProgram, 'get_circuit_async', new_callable=mock.AsyncMock
    ) as get_circuit_async:
        get_circuit_async.return_value = circuit
        assert job.get_circuit() is circuit
        get_circuit_async.assert_called_with(None)
        assert job.get_circuit(1) is circuit
        get_circuit_async.assert_called_with(1)


@duet.sync
async def test_get_circuit_async():
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    circuit = cirq.Circuit()
    with mock.patch.object(
        cg.EngineProgram, 'get_circuit_async', new_callable=mock.AsyncMock
    ) as get_circuit_async:
        get_circuit_async.return_value = circuit
        assert await job.get_circuit_async(1) is circuit
        get_circuit_async.assert_called_with(1)


@mock.patch('cirq_google.engine.engine_program.EngineProgram.is_batch_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_batched_results_non_batch_job_raises(get_job_results, mock_is_batch):
    mock_is_batch.return_value = False
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    with pytest.raises(ValueError, match='batched_results called for a non-batch program'):
        _ = job.batched_results()


@mock.patch('cirq_google.engine.engine_program.EngineProgram.is_batch_async')
@mock.patch('cirq_google.engine.engine_program.EngineProgram.batch_size_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_batched_results_batch_job(get_job_results, mock_batch_size, mock_is_batch):
    mock_is_batch.return_value = True
    mock_batch_size.return_value = 2
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    get_job_results.return_value = RESULTS_NON_UNIFORM
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)

    batched = job.batched_results()
    assert len(batched) == 2
    assert len(batched[0]) == 1
    assert len(batched[1]) == 1
    assert len(batched[0][0].measurements['q']) == 10
    assert len(batched[1][0].measurements['q']) == 20
    assert batched[0][0].job_id == 'steve'
    assert batched[1][0].job_id == 'steve'


@mock.patch('cirq_google.engine.engine_program.EngineProgram.is_batch_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_batched_results_batch_job_v1_raises(get_job_results, mock_is_batch):
    mock_is_batch.return_value = True
    qjob = quantum.QuantumJob(
        execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
        update_time=UPDATE_TIME,
    )
    v1_result = quantum.QuantumResult(result=util.pack_any(v1.program_pb2.Result()))
    get_job_results.return_value = v1_result
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)

    with pytest.raises(ValueError, match='batched_results was not populated for this batch job'):
        _ = job.batched_results()

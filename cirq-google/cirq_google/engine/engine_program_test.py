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

import datetime
from unittest import mock

import numpy as np
import pytest
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge

import cirq.testing
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.cloud import quantum
from cirq_google.engine import util
from cirq_google.engine.engine import EngineContext

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


@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_sweeps_delegation(create_job_async):
    create_job_async.return_value = ('steve', quantum.QuantumJob())
    program = cg.EngineProgram('my-proj', 'my-prog', EngineContext())
    param_resolver = cirq.ParamResolver({})
    job = program.run_sweep(
        job_id='steve',
        repetitions=10,
        params=param_resolver,
        processor_id='mine',
        run_name="run_name",
        device_config_name="config",
    )
    assert job._job == quantum.QuantumJob()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_delegation(create_job_async, get_results_async):
    dt = datetime.datetime.now(tz=datetime.timezone.utc)
    create_job_async.return_value = (
        'steve',
        quantum.QuantumJob(
            name='projects/a/programs/b/jobs/steve',
            execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS),
            update_time=dt,
        ),
    )
    get_results_async.return_value = quantum.QuantumResult(
        result=util.pack_any(
            Merge(
                """sweep_results: [{
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
        }]
    }]
""",
                v2.result_pb2.Result(),
            )
        )
    )

    program = cg.EngineProgram('a', 'b', EngineContext())
    param_resolver = cirq.ParamResolver({})
    results = program.run(
        job_id='steve',
        repetitions=10,
        param_resolver=param_resolver,
        processor_id='mine',
        run_name="run_name",
        device_config_name="config",
    )

    assert results == cg.EngineResult(
        params=cirq.ParamResolver({'a': 1.0}),
        measurements={'q': np.array([[False], [True], [True], [False]], dtype=bool)},
        job_id='steve',
        job_finished_time=dt,
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_jobs_async')
def test_list_jobs(list_jobs_async):
    job1 = quantum.QuantumJob(name='projects/proj/programs/prog1/jobs/job1')
    job2 = quantum.QuantumJob(name='projects/otherproj/programs/prog1/jobs/job2')
    list_jobs_async.return_value = [job1, job2]

    ctx = EngineContext()
    result = cg.EngineProgram(project_id='proj', program_id='prog1', context=ctx).list_jobs()
    list_jobs_async.assert_called_once_with(
        'proj',
        'prog1',
        created_after=None,
        created_before=None,
        has_labels=None,
        execution_states=None,
    )
    assert [(j.program_id, j.project_id, j.job_id, j.context, j._job) for j in result] == [
        ('prog1', 'proj', 'job1', ctx, job1),
        ('prog1', 'otherproj', 'job2', ctx, job2),
    ]


def test_engine():
    program = cg.EngineProgram('a', 'b', EngineContext())
    assert program.engine().project_id == 'a'


def test_get_job():
    program = cg.EngineProgram('a', 'b', EngineContext())
    assert program.get_job('c').job_id == 'c'


def test_create_time():
    program = cg.EngineProgram(
        'a',
        'b',
        EngineContext(),
        _program=quantum.QuantumProgram(create_time=timestamp_pb2.Timestamp(seconds=1581515101)),
    )
    assert program.create_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_update_time(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(
        update_time=timestamp_pb2.Timestamp(seconds=1581515101)
    )
    assert program.update_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )
    get_program_async.assert_called_once_with('a', 'b', False)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_description(get_program_async):
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=quantum.QuantumProgram(description='hello')
    )
    assert program.description() == 'hello'

    get_program_async.return_value = quantum.QuantumProgram(description='hello')
    assert cg.EngineProgram('a', 'b', EngineContext()).description() == 'hello'
    get_program_async.assert_called_once_with('a', 'b', False)


@mock.patch('cirq_google.engine.engine_client.EngineClient.set_program_description_async')
def test_set_description(set_program_description_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    set_program_description_async.return_value = quantum.QuantumProgram(description='world')
    assert program.set_description('world').description() == 'world'
    set_program_description_async.assert_called_with('a', 'b', 'world')

    set_program_description_async.return_value = quantum.QuantumProgram(description='')
    assert program.set_description('').description() == ''
    set_program_description_async.assert_called_with('a', 'b', '')


def test_labels():
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=quantum.QuantumProgram(labels={'t': '1'})
    )
    assert program.labels() == {'t': '1'}


@mock.patch('cirq_google.engine.engine_client.EngineClient.set_program_labels_async')
def test_set_labels(set_program_labels_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    set_program_labels_async.return_value = quantum.QuantumProgram(labels={'a': '1', 'b': '1'})
    assert program.set_labels({'a': '1', 'b': '1'}).labels() == {'a': '1', 'b': '1'}
    set_program_labels_async.assert_called_with('a', 'b', {'a': '1', 'b': '1'})

    set_program_labels_async.return_value = quantum.QuantumProgram()
    assert program.set_labels({}).labels() == {}
    set_program_labels_async.assert_called_with('a', 'b', {})


@mock.patch('cirq_google.engine.engine_client.EngineClient.add_program_labels_async')
def test_add_labels(add_program_labels_async):
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=quantum.QuantumProgram(labels={})
    )
    assert program.labels() == {}

    add_program_labels_async.return_value = quantum.QuantumProgram(labels={'a': '1'})
    assert program.add_labels({'a': '1'}).labels() == {'a': '1'}
    add_program_labels_async.assert_called_with('a', 'b', {'a': '1'})

    add_program_labels_async.return_value = quantum.QuantumProgram(labels={'a': '2', 'b': '1'})
    assert program.add_labels({'a': '2', 'b': '1'}).labels() == {'a': '2', 'b': '1'}
    add_program_labels_async.assert_called_with('a', 'b', {'a': '2', 'b': '1'})


@mock.patch('cirq_google.engine.engine_client.EngineClient.remove_program_labels_async')
def test_remove_labels(remove_program_labels_async):
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=quantum.QuantumProgram(labels={'a': '1', 'b': '1'})
    )
    assert program.labels() == {'a': '1', 'b': '1'}

    remove_program_labels_async.return_value = quantum.QuantumProgram(labels={'b': '1'})
    assert program.remove_labels(['a']).labels() == {'b': '1'}
    remove_program_labels_async.assert_called_with('a', 'b', ['a'])

    remove_program_labels_async.return_value = quantum.QuantumProgram(labels={})
    assert program.remove_labels(['a', 'b', 'c']).labels() == {}
    remove_program_labels_async.assert_called_with('a', 'b', ['a', 'b', 'c'])


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_v1(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(
        code=util.pack_any(v1.program_pb2.Program())
    )

    with pytest.raises(ValueError, match='v1 Program is not supported'):
        program.get_circuit()


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_v2(get_program_async):
    circuit = cirq.Circuit(
        cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
    )

    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(code=_PROGRAM_V2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        program.get_circuit(), circuit
    )
    get_program_async.assert_called_once_with('a', 'b', True)


@pytest.fixture(scope='module', autouse=True)
def mock_grpc_client():
    with mock.patch(
        'cirq_google.engine.engine_client.quantum.QuantumEngineServiceClient'
    ) as _fixture:
        yield _fixture


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_unsupported_program_type(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(
        code=any_pb2.Any(type_url='type.googleapis.com/unknown.proto')
    )

    with pytest.raises(ValueError, match='unknown.proto'):
        program.get_circuit()


@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_program_async')
def test_delete(delete_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    program.delete()
    delete_program_async.assert_called_with('a', 'b', delete_jobs=False)

    program.delete(delete_jobs=True)
    delete_program_async.assert_called_with('a', 'b', delete_jobs=True)


@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_job_async')
def test_delete_jobs(delete_job_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    program.delete_job('c')
    delete_job_async.assert_called_with('a', 'b', 'c')


def test_str():
    program = cg.EngineProgram('my-proj', 'my-prog', EngineContext())
    assert str(program) == 'EngineProgram(project_id=\'my-proj\', program_id=\'my-prog\')'

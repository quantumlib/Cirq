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

import pytest
import numpy as np
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge

import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
from cirq_google.engine.result_type import ResultType
from cirq_google.engine.test_utils import uses_async_mock


_BATCH_PROGRAM_V2 = util.pack_any(
    Merge(
        """programs { language {
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
}
""",
        v2.batch_pb2.BatchProgram(),
    )
)

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


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_sweeps_delegation(create_job_async):
    create_job_async.return_value = ('steve', quantum.QuantumJob())
    program = cg.EngineProgram('my-proj', 'my-prog', EngineContext())
    param_resolver = cirq.ParamResolver({})
    job = program.run_sweep(
        job_id='steve', repetitions=10, params=param_resolver, processor_ids=['mine']
    )
    assert job._job == quantum.QuantumJob()


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_batch_delegation(create_job_async):
    create_job_async.return_value = ('kittens', quantum.QuantumJob())
    program = cg.EngineProgram('my-meow', 'my-meow', EngineContext(), result_type=ResultType.Batch)
    resolver_list = [cirq.Points('cats', [1.0, 2.0, 3.0]), cirq.Points('cats', [4.0, 5.0, 6.0])]
    job = program.run_batch(
        job_id='steve', repetitions=10, params_list=resolver_list, processor_ids=['lazykitty']
    )
    assert job._job == quantum.QuantumJob()


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_calibration_delegation(create_job_async):
    create_job_async.return_value = ('dogs', quantum.QuantumJob())
    program = cg.EngineProgram('woof', 'woof', EngineContext(), result_type=ResultType.Calibration)
    job = program.run_calibration(processor_ids=['lazydog'])
    assert job._job == quantum.QuantumJob()


@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_calibration_no_processors(create_job_async):
    create_job_async.return_value = ('dogs', quantum.QuantumJob())
    program = cg.EngineProgram('woof', 'woof', EngineContext(), result_type=ResultType.Calibration)
    with pytest.raises(ValueError, match='No processors specified'):
        _ = program.run_calibration(job_id='spot')


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.create_job_async')
def test_run_batch_no_sweeps(create_job_async):
    # Running with no sweeps is fine. Uses program's batch size to create
    # proper empty sweeps.
    create_job_async.return_value = ('kittens', quantum.QuantumJob())
    program = cg.EngineProgram(
        'my-meow',
        'my-meow',
        _program=quantum.QuantumProgram(code=_BATCH_PROGRAM_V2),
        context=EngineContext(),
        result_type=ResultType.Batch,
    )
    job = program.run_batch(job_id='steve', repetitions=10, processor_ids=['lazykitty'])
    assert job._job == quantum.QuantumJob()
    batch_run_context = v2.batch_pb2.BatchRunContext()
    create_job_async.call_args[1]['run_context'].Unpack(batch_run_context)
    assert len(batch_run_context.run_contexts) == 1


def test_run_batch_no_processors():
    program = cg.EngineProgram('no-meow', 'no-meow', EngineContext(), result_type=ResultType.Batch)
    resolver_list = [cirq.Points('cats', [1.0, 2.0]), cirq.Points('cats', [3.0, 4.0])]
    with pytest.raises(ValueError, match='No processors specified'):
        _ = program.run_batch(repetitions=1, params_list=resolver_list)


def test_run_batch_not_in_batch_mode():
    program = cg.EngineProgram('no-meow', 'no-meow', EngineContext())
    resolver_list = [cirq.Points('cats', [1.0, 2.0, 3.0]), cirq.Points('cats', [4.0, 5.0, 6.0])]
    with pytest.raises(ValueError, match='Can only use run_batch'):
        _ = program.run_batch(repetitions=1, processor_ids=['lazykitty'], params_list=resolver_list)


def test_run_in_batch_mode():
    program = cg.EngineProgram('no-meow', 'no-meow', EngineContext(), result_type=ResultType.Batch)
    with pytest.raises(ValueError, match='Please use run_batch'):
        _ = program.run_sweep(
            repetitions=1, processor_ids=['lazykitty'], params=cirq.Points('cats', [1.0, 2.0, 3.0])
        )


@uses_async_mock
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
        job_id='steve', repetitions=10, param_resolver=param_resolver, processor_ids=['mine']
    )

    assert results == cg.EngineResult(
        params=cirq.ParamResolver({'a': 1.0}),
        measurements={'q': np.array([[False], [True], [True], [False]], dtype=bool)},
        job_id='steve',
        job_finished_time=dt,
    )


@uses_async_mock
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


@uses_async_mock
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


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_description(get_program_async):
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=quantum.QuantumProgram(description='hello')
    )
    assert program.description() == 'hello'

    get_program_async.return_value = quantum.QuantumProgram(description='hello')
    assert cg.EngineProgram('a', 'b', EngineContext()).description() == 'hello'
    get_program_async.assert_called_once_with('a', 'b', False)


@uses_async_mock
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


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.set_program_labels_async')
def test_set_labels(set_program_labels_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    set_program_labels_async.return_value = quantum.QuantumProgram(labels={'a': '1', 'b': '1'})
    assert program.set_labels({'a': '1', 'b': '1'}).labels() == {'a': '1', 'b': '1'}
    set_program_labels_async.assert_called_with('a', 'b', {'a': '1', 'b': '1'})

    set_program_labels_async.return_value = quantum.QuantumProgram()
    assert program.set_labels({}).labels() == {}
    set_program_labels_async.assert_called_with('a', 'b', {})


@uses_async_mock
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


@uses_async_mock
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


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_v1(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(
        code=util.pack_any(v1.program_pb2.Program())
    )

    with pytest.raises(ValueError, match='v1 Program is not supported'):
        program.get_circuit()


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_v2(get_program_async):
    circuit = cirq.Circuit(
        cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
    )

    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(code=_PROGRAM_V2)
    assert program.get_circuit() == circuit
    get_program_async.assert_called_once_with('a', 'b', True)


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_batch(get_program_async):
    circuit = cirq.Circuit(
        cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
    )

    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(code=_BATCH_PROGRAM_V2)
    with pytest.raises(ValueError, match='A program number must be specified'):
        program.get_circuit()
    with pytest.raises(ValueError, match='Only 1 in the batch but index 1 was specified'):
        program.get_circuit(1)
    assert program.get_circuit(0) == circuit
    get_program_async.assert_called_once_with('a', 'b', True)


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_batch_size(get_program_async):
    # Has to fetch from engine if not _program specified.
    program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Batch)
    get_program_async.return_value = quantum.QuantumProgram(code=_BATCH_PROGRAM_V2)
    assert program.batch_size() == 1

    # If _program specified, uses that value.
    program = cg.EngineProgram(
        'a',
        'b',
        EngineContext(),
        _program=quantum.QuantumProgram(code=_BATCH_PROGRAM_V2),
        result_type=ResultType.Batch,
    )
    assert program.batch_size() == 1

    with pytest.raises(ValueError, match='ResultType.Program'):
        program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Program)
        _ = program.batch_size()

    with pytest.raises(ValueError, match='cirq.google.api.v2.Program'):
        get_program_async.return_value = quantum.QuantumProgram(code=_PROGRAM_V2)
        program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Batch)
        _ = program.batch_size()


@pytest.fixture(scope='session', autouse=True)
def mock_grpc_client():
    with mock.patch(
        'cirq_google.engine.engine_client.quantum.QuantumEngineServiceClient'
    ) as _fixture:
        yield _fixture


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_v2_unknown_gateset(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(
        code=util.pack_any(
            v2.program_pb2.Program(language=v2.program_pb2.Language(gate_set="BAD_GATESET"))
        )
    )

    with pytest.raises(ValueError, match='BAD_GATESET'):
        program.get_circuit()


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_circuit_unsupported_program_type(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(
        code=any_pb2.Any(type_url='type.googleapis.com/unknown.proto')
    )

    with pytest.raises(ValueError, match='unknown.proto'):
        program.get_circuit()


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_program_async')
def test_delete(delete_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    program.delete()
    delete_program_async.assert_called_with('a', 'b', delete_jobs=False)

    program.delete(delete_jobs=True)
    delete_program_async.assert_called_with('a', 'b', delete_jobs=True)


@uses_async_mock
@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_job_async')
def test_delete_jobs(delete_job_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    program.delete_job('c')
    delete_job_async.assert_called_with('a', 'b', 'c')


def test_str():
    program = cg.EngineProgram('my-proj', 'my-prog', EngineContext())
    assert str(program) == 'EngineProgram(project_id=\'my-proj\', program_id=\'my-prog\')'

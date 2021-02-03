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

from unittest import mock

import datetime
import pytest
import numpy as np
from google.protobuf.text_format import Merge
import cirq
import cirq.google as cg
from cirq.google.api import v1, v2
from cirq.google.engine.client.quantum_v1alpha1 import types as qtypes
from cirq.google.engine.engine import EngineContext
from cirq.google.engine.result_type import ResultType


def _to_any(proto):
    any_proto = qtypes.any_pb2.Any()
    any_proto.Pack(proto)
    return any_proto


_BATCH_PROGRAM_V2 = _to_any(
    Merge(
        """programs { language {
  gate_set: "xmon"
}
circuit {
  scheduling_strategy: MOMENT_BY_MOMENT
  moments {
    operations {
      gate {
        id: "xy"
      }
      args {
        key: "axis_half_turns"
        value {
          arg_value {
            float_value: 0.0
          }
        }
      }
      args {
        key: "half_turns"
        value {
          arg_value {
            float_value: 0.5
          }
        }
      }
      qubits {
        id: "5_2"
      }
    }
  }
  moments {
    operations {
      gate {
        id: "meas"
      }
      args {
        key: "invert_mask"
        value {
          arg_value {
            bool_values {
            }
          }
        }
      }
      args {
        key: "key"
        value {
          arg_value {
            string_value: "result"
          }
        }
      }
      qubits {
        id: "5_2"
      }
    }
  }
}
}
""",
        v2.batch_pb2.BatchProgram(),
    )
)

_PROGRAM_V2 = _to_any(
    Merge(
        """language {
  gate_set: "xmon"
}
circuit {
  scheduling_strategy: MOMENT_BY_MOMENT
  moments {
    operations {
      gate {
        id: "xy"
      }
      args {
        key: "axis_half_turns"
        value {
          arg_value {
            float_value: 0.0
          }
        }
      }
      args {
        key: "half_turns"
        value {
          arg_value {
            float_value: 0.5
          }
        }
      }
      qubits {
        id: "5_2"
      }
    }
  }
  moments {
    operations {
      gate {
        id: "meas"
      }
      args {
        key: "invert_mask"
        value {
          arg_value {
            bool_values {
            }
          }
        }
      }
      args {
        key: "key"
        value {
          arg_value {
            string_value: "result"
          }
        }
      }
      qubits {
        id: "5_2"
      }
    }
  }
}
""",
        v2.program_pb2.Program(),
    )
)


@mock.patch('cirq.google.engine.engine_client.EngineClient.create_job')
def test_run_sweeps_delegation(create_job):
    create_job.return_value = ('steve', qtypes.QuantumJob())
    program = cg.EngineProgram('my-proj', 'my-prog', EngineContext())
    param_resolver = cirq.ParamResolver({})
    job = program.run_sweep(
        job_id='steve', repetitions=10, params=param_resolver, processor_ids=['mine']
    )
    assert job._job == qtypes.QuantumJob()


@mock.patch('cirq.google.engine.engine_client.EngineClient.create_job')
def test_run_batch_delegation(create_job):
    create_job.return_value = ('kittens', qtypes.QuantumJob())
    program = cg.EngineProgram('my-meow', 'my-meow', EngineContext(), result_type=ResultType.Batch)
    resolver_list = [cirq.Points('cats', [1.0, 2.0, 3.0]), cirq.Points('cats', [4.0, 5.0, 6.0])]
    job = program.run_batch(
        job_id='steve', repetitions=10, params_list=resolver_list, processor_ids=['lazykitty']
    )
    assert job._job == qtypes.QuantumJob()


@mock.patch('cirq.google.engine.engine_client.EngineClient.create_job')
def test_run_calibration_delegation(create_job):
    create_job.return_value = ('dogs', qtypes.QuantumJob())
    program = cg.EngineProgram('woof', 'woof', EngineContext(), result_type=ResultType.Calibration)
    job = program.run_calibration(processor_ids=['lazydog'])
    assert job._job == qtypes.QuantumJob()


@mock.patch('cirq.google.engine.engine_client.EngineClient.create_job')
def test_run_calibration_no_processors(create_job):
    create_job.return_value = ('dogs', qtypes.QuantumJob())
    program = cg.EngineProgram('woof', 'woof', EngineContext(), result_type=ResultType.Calibration)
    with pytest.raises(ValueError, match='No processors specified'):
        _ = program.run_calibration(job_id='spot')


@mock.patch('cirq.google.engine.engine_client.EngineClient.create_job')
def test_run_batch_no_sweeps(create_job):
    # Running with no sweeps is fine. Uses program's batch size to create
    # proper empty sweeps.
    create_job.return_value = ('kittens', qtypes.QuantumJob())
    program = cg.EngineProgram(
        'my-meow',
        'my-meow',
        _program=qtypes.QuantumProgram(code=_BATCH_PROGRAM_V2),
        context=EngineContext(),
        result_type=ResultType.Batch,
    )
    job = program.run_batch(job_id='steve', repetitions=10, processor_ids=['lazykitty'])
    assert job._job == qtypes.QuantumJob()
    batch_run_context = v2.batch_pb2.BatchRunContext()
    create_job.call_args[1]['run_context'].Unpack(batch_run_context)
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


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_job_results')
@mock.patch('cirq.google.engine.engine_client.EngineClient.create_job')
def test_run_delegation(create_job, get_results):
    create_job.return_value = (
        'steve',
        qtypes.QuantumJob(
            name='projects/a/programs/b/jobs/steve',
            execution_status=qtypes.ExecutionStatus(state=qtypes.ExecutionStatus.State.SUCCESS),
        ),
    )
    get_results.return_value = qtypes.QuantumResult(
        result=_to_any(
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

    assert results == cirq.Result(
        params=cirq.ParamResolver({'a': 1.0}),
        measurements={'q': np.array([[False], [True], [True], [False]], dtype=np.bool)},
    )


@mock.patch('cirq.google.engine.engine_client.EngineClient.list_jobs')
def test_list_jobs(list_jobs):
    job1 = qtypes.QuantumJob(name='projects/proj/programs/prog1/jobs/job1')
    job2 = qtypes.QuantumJob(name='projects/otherproj/programs/prog1/jobs/job2')
    list_jobs.return_value = [job1, job2]

    ctx = EngineContext()
    result = cg.EngineProgram(project_id='proj', program_id='prog1', context=ctx).list_jobs()
    list_jobs.assert_called_once_with(
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
        _program=qtypes.QuantumProgram(
            create_time=qtypes.timestamp_pb2.Timestamp(seconds=1581515101)
        ),
    )
    assert program.create_time() == datetime.datetime(2020, 2, 12, 13, 45, 1)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_update_time(get_program):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program.return_value = qtypes.QuantumProgram(
        update_time=qtypes.timestamp_pb2.Timestamp(seconds=1581515101)
    )
    assert program.update_time() == datetime.datetime(2020, 2, 12, 13, 45, 1)
    get_program.assert_called_once_with('a', 'b', False)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_description(get_program):
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=qtypes.QuantumProgram(description='hello')
    )
    assert program.description() == 'hello'

    get_program.return_value = qtypes.QuantumProgram(description='hello')
    assert cg.EngineProgram('a', 'b', EngineContext()).description() == 'hello'
    get_program.assert_called_once_with('a', 'b', False)


@mock.patch('cirq.google.engine.engine_client.EngineClient.set_program_description')
def test_set_description(set_program_description):
    program = cg.EngineProgram('a', 'b', EngineContext())
    set_program_description.return_value = qtypes.QuantumProgram(description='world')
    assert program.set_description('world').description() == 'world'
    set_program_description.assert_called_with('a', 'b', 'world')

    set_program_description.return_value = qtypes.QuantumProgram(description='')
    assert program.set_description('').description() == ''
    set_program_description.assert_called_with('a', 'b', '')


def test_labels():
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=qtypes.QuantumProgram(labels={'t': '1'})
    )
    assert program.labels() == {'t': '1'}


@mock.patch('cirq.google.engine.engine_client.EngineClient.set_program_labels')
def test_set_labels(set_program_labels):
    program = cg.EngineProgram('a', 'b', EngineContext())
    set_program_labels.return_value = qtypes.QuantumProgram(labels={'a': '1', 'b': '1'})
    assert program.set_labels({'a': '1', 'b': '1'}).labels() == {'a': '1', 'b': '1'}
    set_program_labels.assert_called_with('a', 'b', {'a': '1', 'b': '1'})

    set_program_labels.return_value = qtypes.QuantumProgram()
    assert program.set_labels({}).labels() == {}
    set_program_labels.assert_called_with('a', 'b', {})


@mock.patch('cirq.google.engine.engine_client.EngineClient.add_program_labels')
def test_add_labels(add_program_labels):
    program = cg.EngineProgram('a', 'b', EngineContext(), _program=qtypes.QuantumProgram(labels={}))
    assert program.labels() == {}

    add_program_labels.return_value = qtypes.QuantumProgram(
        labels={
            'a': '1',
        }
    )
    assert program.add_labels({'a': '1'}).labels() == {'a': '1'}
    add_program_labels.assert_called_with('a', 'b', {'a': '1'})

    add_program_labels.return_value = qtypes.QuantumProgram(labels={'a': '2', 'b': '1'})
    assert program.add_labels({'a': '2', 'b': '1'}).labels() == {'a': '2', 'b': '1'}
    add_program_labels.assert_called_with('a', 'b', {'a': '2', 'b': '1'})


@mock.patch('cirq.google.engine.engine_client.EngineClient.remove_program_labels')
def test_remove_labels(remove_program_labels):
    program = cg.EngineProgram(
        'a', 'b', EngineContext(), _program=qtypes.QuantumProgram(labels={'a': '1', 'b': '1'})
    )
    assert program.labels() == {'a': '1', 'b': '1'}

    remove_program_labels.return_value = qtypes.QuantumProgram(
        labels={
            'b': '1',
        }
    )
    assert program.remove_labels(['a']).labels() == {'b': '1'}
    remove_program_labels.assert_called_with('a', 'b', ['a'])

    remove_program_labels.return_value = qtypes.QuantumProgram(labels={})
    assert program.remove_labels(['a', 'b', 'c']).labels() == {}
    remove_program_labels.assert_called_with('a', 'b', ['a', 'b', 'c'])


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_get_circuit_v1(get_program):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program.return_value = qtypes.QuantumProgram(code=_to_any(v1.program_pb2.Program()))

    with pytest.raises(ValueError, match='v1 Program is not supported'):
        program.get_circuit()


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_get_circuit_v2(get_program):
    circuit = cirq.Circuit(
        cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
    )

    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program.return_value = qtypes.QuantumProgram(code=_PROGRAM_V2)
    assert program.get_circuit() == circuit
    get_program.assert_called_once_with('a', 'b', True)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_get_circuit_batch(get_program):
    circuit = cirq.Circuit(
        cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
    )

    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program.return_value = qtypes.QuantumProgram(code=_BATCH_PROGRAM_V2)
    with pytest.raises(ValueError, match='A program number must be specified'):
        program.get_circuit()
    with pytest.raises(ValueError, match='Only 1 in the batch but index 1 was specified'):
        program.get_circuit(1)
    assert program.get_circuit(0) == circuit
    get_program.assert_called_once_with('a', 'b', True)


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_get_batch_size(get_program):
    # Has to fetch from engine if not _program specified.
    program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Batch)
    get_program.return_value = qtypes.QuantumProgram(code=_BATCH_PROGRAM_V2)
    assert program.batch_size() == 1

    # If _program specified, uses that value.
    program = cg.EngineProgram(
        'a',
        'b',
        EngineContext(),
        _program=qtypes.QuantumProgram(code=_BATCH_PROGRAM_V2),
        result_type=ResultType.Batch,
    )
    assert program.batch_size() == 1

    with pytest.raises(ValueError, match='ResultType.Program'):
        program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Program)
        _ = program.batch_size()

    with pytest.raises(ValueError, match='cirq.google.api.v2.Program'):
        get_program.return_value = qtypes.QuantumProgram(code=_PROGRAM_V2)
        program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Batch)
        _ = program.batch_size()


@pytest.fixture(scope='session', autouse=True)
def mock_grpc_client():
    with mock.patch(
        'cirq.google.engine.engine_client.quantum.QuantumEngineServiceClient'
    ) as _fixture:
        yield _fixture


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_get_circuit_v2_unknown_gateset(get_program):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program.return_value = qtypes.QuantumProgram(
        code=_to_any(
            v2.program_pb2.Program(language=v2.program_pb2.Language(gate_set="BAD_GATESET"))
        )
    )

    with pytest.raises(ValueError, match='BAD_GATESET'):
        program.get_circuit()


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_program')
def test_get_circuit_unsupported_program_type(get_program):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program.return_value = qtypes.QuantumProgram(
        code=qtypes.any_pb2.Any(type_url='type.googleapis.com/unknown.proto')
    )

    with pytest.raises(ValueError, match='unknown.proto'):
        program.get_circuit()


@mock.patch('cirq.google.engine.engine_client.EngineClient.delete_program')
def test_delete(delete_program):
    program = cg.EngineProgram('a', 'b', EngineContext())
    program.delete()
    delete_program.assert_called_with('a', 'b', delete_jobs=False)

    program.delete(delete_jobs=True)
    delete_program.assert_called_with('a', 'b', delete_jobs=True)


def test_str():
    program = cg.EngineProgram('my-proj', 'my-prog', EngineContext())
    assert str(program) == 'EngineProgram(project_id=\'my-proj\', program_id=\'my-prog\')'

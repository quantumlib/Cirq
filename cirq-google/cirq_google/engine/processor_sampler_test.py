# Copyright 2022 The Cirq Developers
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

import pytest

import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor


@pytest.mark.parametrize('circuit', [cirq.Circuit(), cirq.FrozenCircuit()])
@pytest.mark.parametrize(
    'run_name, device_config_name', [('run_name', 'device_config_alias'), ('', '')]
)
def test_run_circuit(circuit, run_name, device_config_name):
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(
        processor=processor, run_name=run_name, device_config_name=device_config_name
    )
    params = [cirq.ParamResolver({'a': 1})]
    sampler.run_sweep(circuit, params, 5)
    processor.run_sweep_async.assert_called_with(
        params=params,
        program=circuit,
        repetitions=5,
        run_name=run_name,
        device_config_name=device_config_name,
    )


@pytest.mark.parametrize(
    'run_name, device_config_name', [('run_name', 'device_config_alias'), ('', '')]
)
def test_run_batch(run_name, device_config_name):
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(
        processor=processor, run_name=run_name, device_config_name=device_config_name
    )
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    sampler.run_batch(circuits, params_list, 5)
    processor.run_batch_async.assert_called_with(
        params_list=params_list,
        programs=circuits,
        repetitions=5,
        run_name=run_name,
        device_config_name=device_config_name,
    )


@pytest.mark.parametrize(
    'run_name, device_config_name', [('run_name', 'device_config_alias'), ('', '')]
)
def test_run_batch_identical_repetitions(run_name, device_config_name):
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(
        processor=processor, run_name=run_name, device_config_name=device_config_name
    )
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    sampler.run_batch(circuits, params_list, [5, 5])
    processor.run_batch_async.assert_called_with(
        params_list=params_list,
        programs=circuits,
        repetitions=5,
        run_name=run_name,
        device_config_name=device_config_name,
    )


def test_run_batch_bad_number_of_repetitions():
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(processor=processor)
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    with pytest.raises(ValueError, match='2 and 3'):
        sampler.run_batch(circuits, params_list, [5, 5, 5])


def test_run_batch_differing_repetitions():
    processor = mock.create_autospec(AbstractProcessor)
    run_name = "RUN_NAME"
    device_config_name = "DEVICE_CONFIG_NAME"
    sampler = cg.ProcessorSampler(
        processor=processor, run_name=run_name, device_config_name=device_config_name
    )
    job = mock.Mock()
    job.results.return_value = []
    processor.run_sweep.return_value = job
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    repetitions = [1, 2]
    sampler.run_batch(circuits, params_list, repetitions)
    processor.run_sweep_async.assert_called_with(
        params=params2,
        program=circuit2,
        repetitions=2,
        run_name=run_name,
        device_config_name=device_config_name,
    )
    processor.run_batch_async.assert_not_called()


def test_processor_sampler_processor_property():
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(processor=processor)
    assert sampler.processor is processor


def test_with_local_processor():
    sampler = cg.ProcessorSampler(
        processor=cg.engine.SimulatedLocalProcessor(processor_id='my-fancy-processor')
    )
    r = sampler.run(cirq.Circuit(cirq.measure(cirq.LineQubit(0), key='z')))
    assert isinstance(r, cg.EngineResult)
    assert r.job_id == 'projects/fake_project/processors/my-fancy-processor/job/2'
    assert r.measurements['z'] == [[0]]


@pytest.mark.parametrize(
    'run_name, device_config_name', [('run_name', ''), ('', 'device_config_name')]
)
def test_processor_sampler_with_invalid_configuration_throws(run_name, device_config_name):
    processor = mock.create_autospec(AbstractProcessor)
    with pytest.raises(
        ValueError, match='Cannot specify only one of `run_name` and `device_config_name`'
    ):
        cg.ProcessorSampler(
            processor=processor, run_name=run_name, device_config_name=device_config_name
        )

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


class _DummyProcessorRecord(cg.ProcessorRecord):
    def get_processor(self) -> 'cg.engine.AbstractProcessor':
        return cg.engine.SimulatedLocalProcessor(processor_id='dummy')


def test_abstract_processor_record():
    proc_rec = _DummyProcessorRecord()
    assert isinstance(proc_rec.get_processor(), cg.engine.AbstractProcessor)
    assert isinstance(proc_rec.get_sampler(), cirq.Sampler)
    assert isinstance(proc_rec.get_device(), cirq.Device)


def _set_get_processor_return(get_processor):
    # from engine_test.py

    from google.protobuf.text_format import Merge

    from cirq_google.api import v2
    from cirq_google.engine import util
    from cirq_google.engine.client.quantum_v1alpha1 import types as qtypes

    device_spec = util.pack_any(
        Merge(
            """
valid_gate_sets: [{
    name: 'test_set',
    valid_gates: [{
        id: 'x',
        number_of_qubits: 1,
        gate_duration_picos: 1000,
        valid_targets: ['1q_targets']
    }]
}],
valid_qubits: ['0_0', '1_1'],
valid_targets: [{
    name: '1q_targets',
    target_ordering: SYMMETRIC,
    targets: [{
        ids: ['0_0']
    }]
}]
""",
            v2.device_pb2.DeviceSpecification(),
        )
    )

    get_processor.return_value = qtypes.QuantumProcessor(device_spec=device_spec)
    return get_processor


@mock.patch('cirq_google.engine.client.quantum.QuantumEngineServiceClient')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor')
def test_engine_backend(get_processor, _):
    _set_get_processor_return(get_processor)

    with mock.patch('google.auth.default', lambda: (None, 'project!')):
        proc_rec = cg.EngineProcessorRecord('rainbow')
        assert proc_rec.processor_id == 'rainbow'
        assert isinstance(proc_rec.get_processor(), cg.engine.AbstractProcessor)
        assert isinstance(proc_rec.get_sampler(), cirq.Sampler)
        assert isinstance(proc_rec.get_device(), cirq.Device)
    cirq.testing.assert_equivalent_repr(proc_rec, global_vals={'cirq_google': cg})
    assert str(proc_rec) == 'rainbow'


@mock.patch('cirq_google.engine.client.quantum.QuantumEngineServiceClient')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor')
def test_simulated_backend(get_processor, _):
    _set_get_processor_return(get_processor)
    with mock.patch('google.auth.default', lambda: (None, 'project!')):
        proc_rec = cg.SimulatedProcessorRecord('rainbow')
        assert isinstance(proc_rec.get_processor(), cg.engine.AbstractProcessor)
        assert isinstance(proc_rec.get_sampler(), cirq.Sampler)
        assert isinstance(proc_rec.get_device(), cirq.Device)

    assert proc_rec.processor_id == 'rainbow'
    assert str(proc_rec) == 'rainbow-simulator'
    cirq.testing.assert_equivalent_repr(proc_rec, global_vals={'cirq_google': cg})


def test_simulated_backend_with_local_device():
    proc_rec = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    assert isinstance(proc_rec.get_processor(), cg.engine.AbstractProcessor)
    assert proc_rec.processor_id == 'rainbow'
    assert str(proc_rec) == 'rainbow-simulator'

    cirq.testing.assert_equivalent_repr(proc_rec, global_vals={'cirq_google': cg})


def test_simulated_backend_with_bad_local_device():
    proc_rec = cg.SimulatedProcessorWithLocalDeviceRecord('my_processor')
    with pytest.raises(KeyError):
        proc_rec.get_device()


def test_simulated_backend_descriptive_name():
    p = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    assert str(p) == 'rainbow-simulator'
    assert isinstance(p.get_sampler(), cg.ValidatingSampler)
    assert isinstance(p.get_sampler()._sampler, cirq.Simulator)

    p = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow', noise_strength=1e-3)
    assert str(p) == 'rainbow-depol(1.000e-03)'
    assert isinstance(p.get_sampler()._sampler, cirq.DensityMatrixSimulator)

    p = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow', noise_strength=float('inf'))
    assert str(p) == 'rainbow-zeros'
    assert isinstance(p.get_sampler()._sampler, cirq.ZerosSampler)


def test_sampler_equality():
    p = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    assert p.get_sampler().__class__ == p.get_processor().get_sampler().__class__

# Copyright 2018 The Cirq Developers
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

"""Tests for processor_config classes."""

from __future__ import annotations

import cirq_google as cg
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.devices import GridDevice
from cirq_google.engine import util
from cirq_google.engine.processor_config import Run

_METRIC_SNAPSHOT = v2.metrics_pb2.MetricsSnapshot(
    timestamp_ms=1562544000021,
    metrics=[
        v2.metrics_pb2.Metric(
            name='xeb', targets=['0_0', '0_1'], values=[v2.metrics_pb2.Value(double_val=0.9999)]
        ),
        v2.metrics_pb2.Metric(
            name='xeb', targets=['0_0', '1_0'], values=[v2.metrics_pb2.Value(double_val=0.9998)]
        ),
    ],
)

_DEVICE_SPEC = v2.device_pb2.DeviceSpecification(
    valid_qubits=["0_0", "1_1", "2_2"],
    valid_targets=[
        v2.device_pb2.TargetSet(
            name="2_quibit_targets",
            target_ordering=v2.device_pb2.TargetSet.SYMMETRIC,
            targets=[v2.device_pb2.Target(ids=["0_0", "1_1"])],
        )
    ],
)

_PROCESSOR_ID = 'test_processor_id'
_PROJECT_ID = 'test_project_id'
_SNAPSHOT_ID = 'test_snapshot_id'
_CONFIG_NAME = 'test_config_name'

_VALID_QUANTUM_PROCESSOR_CONFIG = quantum.QuantumProcessorConfig(
    name=f'projects/{_PROJECT_ID}/processors/{_PROCESSOR_ID}/configSnapshots/{_SNAPSHOT_ID}/configs/{_CONFIG_NAME}',
    device_specification=util.pack_any(_DEVICE_SPEC),
    characterization=util.pack_any(_METRIC_SNAPSHOT),
)


def test_processor_config_snapshot_id():
    config = cg.engine.ProcessorConfig(
        processor=None, quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG
    )

    assert config.snapshot_id == _SNAPSHOT_ID


def test_processor_config_snapshot_id_empty():
    quantum_config = quantum.QuantumProcessorConfig(
        name='projects/proj_id/processors/proc_id/configAutomationRuns/default/configs/default',
        device_specification=util.pack_any(_DEVICE_SPEC),
        characterization=util.pack_any(_METRIC_SNAPSHOT),
    )
    config = cg.engine.ProcessorConfig(processor=None, quantum_processor_config=quantum_config)

    assert config.snapshot_id == ''


def test_processor_config_run_name():
    run = Run(id='test_run_name')
    config = cg.engine.ProcessorConfig(
        processor=None,
        quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        device_config_revision=run,
    )

    assert config.run_name == run.id


def test_processor_config_effective_device():
    config = cg.engine.ProcessorConfig(
        processor=None, quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG
    )

    assert config.effective_device == GridDevice.from_proto(_DEVICE_SPEC)


def test_processor_config_calibration():
    config = cg.engine.ProcessorConfig(
        processor=None, quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG
    )

    assert config.calibration == cg.Calibration(_METRIC_SNAPSHOT)


def test_processor_processor_id():
    config = cg.engine.ProcessorConfig(
        processor=None, quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG
    )

    assert config.processor_id == _PROCESSOR_ID


def test_processor_config_name():
    config = cg.engine.ProcessorConfig(
        processor=None, quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG
    )

    assert config.config_name == _CONFIG_NAME


def test_processor_config_repr():
    config = cg.engine.ProcessorConfig(
        processor=None, quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG
    )
    expected_repr = (
        'cirq_google.ProcessorConfig'
        f'processor_id={_PROCESSOR_ID}, '
        f'snapshot_id={_SNAPSHOT_ID}, '
        f'run_name={""} '
        f'config_name={_CONFIG_NAME}'
    )

    assert repr(config) == expected_repr


def test_processor_config_repr_with_run_name():
    run = Run(id='test_run_name')
    config = cg.engine.ProcessorConfig(
        processor=None,
        quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        device_config_revision=run,
    )
    expected_repr = (
        'cirq_google.ProcessorConfig'
        f'processor_id={_PROCESSOR_ID}, '
        f'snapshot_id={_SNAPSHOT_ID}, '
        f'run_name={run.id} '
        f'config_name={_CONFIG_NAME}'
    )

    assert repr(config) == expected_repr


def test_sampler():
    run = Run(id='test_run_name')
    config = cg.engine.ProcessorConfig(
        processor=None,
        quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        device_config_revision=run,
    )
    sampler = config.sampler()

    assert sampler.run_name == run.id
    assert sampler.snapshot_id == _SNAPSHOT_ID
    assert sampler.device_config_name == _CONFIG_NAME

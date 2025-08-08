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

import pytest

from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.devices import GridDevice
from cirq_google.engine import util

from google.protobuf import any_pb2

_METRIC_SNAPSHOT = v2.metrics_pb2.MetricsSnapshot(
            timestamp_ms=1562544000021,
            metrics=[
                v2.metrics_pb2.Metric(
                    name='xeb',
                    targets=['0_0', '0_1'],
                    values=[v2.metrics_pb2.Value(double_val=0.9999)],
                ),
                v2.metrics_pb2.Metric(
                    name='xeb',
                    targets=['0_0', '1_0'],
                    values=[v2.metrics_pb2.Value(double_val=0.9998)],
                ),
            ],
        )


_DEVICE_SPEC = v2.device_pb2.DeviceSpecification(
    valid_qubits=["0_0", "1_1", "2_2"],
    valid_targets=[
        v2.device_pb2.TargetSet(
            name="2_quibit_targets",
            target_ordering=v2.device_pb2.TargetSet.SYMMETRIC,
            targets=[v2.device_pb2.Target(
                ids=["0_0", "1_1"]
            )]
        )
    ],
)

_PROCESSOR_ID = 'test_processor_id'
_PROJECT_ID = 'test_project_id'
_SNAPSHOT_ID = 'test_snapshot_id'
_CONFIG_ID = 'test_config_id'

_VALID_QUANTUM_PROCESSOR_CONFIG = quantum.QuantumProcessorConfig(
        name=f'projects/{_PROJECT_ID}/processors/{_PROCESSOR_ID}/configSnapshots/{_SNAPSHOT_ID}/configs/{_CONFIG_ID}',
        device_specification=util.pack_any(_DEVICE_SPEC),
        characterization=util.pack_any(_METRIC_SNAPSHOT)
    )

def test_processor_config_snapshot_id():
    config = cg.engine.ProcessorConfig(
            quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG
        )
    
    assert config.snapshot_id == _SNAPSHOT_ID

def test_processor_config_run_name():    
    run_name = 'test_run_name'
    config = cg.engine.ProcessorConfig(
            quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
            run_name=run_name
        )
    
    assert config.run_name == run_name

def test_processor_config_effective_device():
    config = cg.engine.ProcessorConfig(
            quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        )
    
    assert config.effective_device == GridDevice.from_proto(_DEVICE_SPEC)

def test_processor_config_calibration(): 
    config = cg.engine.ProcessorConfig(
            quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        )
    
    assert config.calibration == cg.Calibration(_METRIC_SNAPSHOT)

def test_processor_project_id(): 
    config = cg.engine.ProcessorConfig(
            quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        )
    
    assert config.project_id == _PROJECT_ID

def test_processor_processor_id(): 
    config = cg.engine.ProcessorConfig(
            quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        )
    
    assert config.processor_id == _PROCESSOR_ID

def test_processor_config_id(): 
    config = cg.engine.ProcessorConfig(
            quantum_processor_config=_VALID_QUANTUM_PROCESSOR_CONFIG,
        )
    
    assert config.config_id == _CONFIG_ID


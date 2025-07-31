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

from unittest import mock
import datetime
import pytest

from google.protobuf.text_format import Merge
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.devices import GridDevice
from cirq_google.engine import engine_client, util
from cirq_google.engine.engine import EngineContext

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


@pytest.fixture(scope='module', autouse=True)
def mock_grpc_client():
    with mock.patch(
        'cirq_google.engine.engine_client.quantum.QuantumEngineServiceClient'
    ) as _fixture:
        yield _fixture

@mock.patch('cirq_google.engine.engine_client.EngineClient.get_quantum_processor_config_by_snapshot_id_async')
def test_get_config(get_quantum_processor):
    project_id = "test_project_id"
    processor_id = "test_proc_id"
    snapshot_id = "test_proc_id"
    config_id = "test_config_id"
    name = f'projects/{project_id}/processors/{processor_id}/configSnapshots/{snapshot_id}/configs/{config_id}'
    snapshot = cg.engine.ProcessorConfigSnapshot(
        project_id=project_id, processor_id=processor_id,
        snapshot_id=snapshot_id, context=EngineContext()
    )

    expected_config = cg.engine.ProcessorConfig(
        name=name,
        effective_device=GridDevice.from_proto(_DEVICE_SPEC),
        calibration=cg.Calibration(_METRIC_SNAPSHOT)
    )   

    quantum_config = quantum.QuantumProcessorConfig(
        name=name,
        device_specification=util.pack_any(_DEVICE_SPEC),
        characterization=util.pack_any(_METRIC_SNAPSHOT)
    )
    get_quantum_processor.return_value = quantum_config
    
    actual_config = snapshot.get_config(config_id)

    get_quantum_processor.assert_called_once_with(
        project_id=project_id, processor_id=processor_id,
        snapshot_id=snapshot_id, config_id=config_id
    )
    assert actual_config.name == expected_config.name
    assert actual_config.effective_device == expected_config.effective_device
    assert actual_config.calibration == expected_config.calibration

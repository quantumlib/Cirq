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

from __future__ import annotations

from typing import TYPE_CHECKING

import cirq_google as cg
from cirq_google.api import v2
from cirq_google.engine import util

if TYPE_CHECKING:
    import cirq
    from cirq_google.cloud.quantum_v1alpha1.types import quantum


class ProcessorConfig:
    """Representation of a quantum processor configuration.

    Describes available qubits, gates, and calibration data associated with
    a processor configuration.
    """

    def __init__(
        self, *, quantum_processor_config: quantum.QuantumProcessorConfig, run_name: str = ''
    ) -> None:
        """Contructs a Processor Config.

        Args:
            quantum_processor_config: The quantum processor config.
        """
        self._quantum_processor_config = quantum_processor_config
        self._run_name = run_name
        self._grid_device = cg.GridDevice.from_proto(
            util.unpack_any(
                self._quantum_processor_config.device_specification,
                v2.device_pb2.DeviceSpecification(),
            )
        )
        self._calibration = cg.Calibration(
            util.unpack_any(
                self._quantum_processor_config.characterization, v2.metrics_pb2.MetricsSnapshot()
            )
        )

    @property
    def effective_device(self) -> cirq.Device:
        """The GridDevice generated from thisc configuration's device specification"""
        return self._grid_device

    @property
    def calibration(self) -> cg.Calibration:
        """Charicterization metrics captured for this configuration"""
        return self._calibration

    @property
    def snapshot_id(self) -> str:
        """The snapshot that contains this processor config"""
        if 'configSnapshots' not in self._quantum_processor_config.name:
            # We assume the calling `get_quantume_processor_config` always
            # returns a config with the snapshot resouce nanme.  This check
            # is added in case this behavior changes in the future.
            return ''
        parts = self._quantum_processor_config.name.split('/')
        return parts[5]

    @property
    def run_name(self) -> str:
        """The run that generated this config if avaiable."""
        return self._run_name

    @property
    def processor_id(self) -> str:
        """The processor id for this config."""
        parts = self._quantum_processor_config.name.split('/')
        return parts[3]

    @property
    def config_name(self) -> str:
        """The unique identifier for this config."""
        parts = self._quantum_processor_config.name.split('/')
        return parts[-1]

    def __repr__(self) -> str:
        return (
            'cirq_google.ProcessorConfig'
            f'processor_id={self.processor_id}, '
            f'snapshot_id={self.snapshot_id}, '
            f'run_name={self.run_name} '
            f'config_name={self.config_name}'
        )

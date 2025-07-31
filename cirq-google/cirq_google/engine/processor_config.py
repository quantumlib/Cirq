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

from cirq_google.devices import GridDevice
from cirq_google.api import v2
from cirq_google.cloud.quantum_v1alpha1.types import quantum


from cirq_google.engine import util
import cirq_google as cg

if TYPE_CHECKING:
    import cirq_google.engine.engine as engine_base

class ProcessorConfig:
    """Representation of a quantum processor configuration

    Describes available qubits, gates, and calivration data associated with
    a processor configuration.
    """

    def __init__(self,
                 *,
                 name: str,
                 effective_device: GridDevice,
                 calibration: cg.Calibration,
    ) -> None:
        self._name = name
        self._effective_device = effective_device
        self._calibration = calibration
    
    @classmethod
    def from_quantum_config(
        cls, quantum_config: quantum.QuantumProcessorConfig
    ) -> ProcessorConfig:
        """Create instance from a QuantumProcessorConfig

        Args:
            quantum_config: The `QuantumProcessorConfig` to create.
        
        Raises:
            ValueError: If the quantum_config.device_specification is invalid
        
        Returns:
            The ProcessorConfig
        """
        name = quantum_config.name
        device_spec = util.unpack_any(
            quantum_config.device_specification, v2.device_pb2.DeviceSpecification()
        )
        characterization = util.unpack_any(
            quantum_config.characterization, v2.metrics_pb2.MetricsSnapshot()
        )
        
        return ProcessorConfig(
            name=name,
            effective_device=cg.GridDevice.from_proto(device_spec),
            calibration=cg.Calibration(characterization)
        )

    @property
    def name(self) -> str:
        """The name of this configuration"""
        return self._name
    
    @property
    def effective_device(self) -> GridDevice:
        """The GridDevice generated from this configuration's device specification"""
        return self._effective_device

    @property
    def calibration(self) -> cg.Calibration:
        """Charicterization metrics captured for this configuration"""
        return self._calibration


class ProcessorConfigSnapshot:
    """A snapshot of available device configurations for a processor."""

    def __init__(self,
                 *,
                 project_id: str,
                 processor_id: str,
                 snapshot_id: str,
                 context: engine_base.EngineContext,
    ) -> None:
        self._project_id = project_id
        self._processor_id = processor_id
        self._snapshot_id = snapshot_id
        self._context = context
    
    @property
    def snapshot_id(self) -> str:
        """The indentifier for this snapshot."""
        return self._snapshot_id
    
    def get_config(self, name: str) -> ProcessorConfig | None:
        """Returns the configuration with the given name in this snapshot if it exists.

        Args:
            name: The name of the configuration.

        Raises:
            ValueError: If config is invalid.
        """
        response = self._context.client.get_quantum_processor_config_by_snapshot_id(
            project_id=self._project_id, processor_id=self._processor_id,
            snapshot_id=self._snapshot_id, config_id=name
        )    
        return ProcessorConfig.from_quantum_config(response)


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

import datetime

if TYPE_CHECKING:
    import cirq_google as cg

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
                 snapshot_id: str,
                 create_time: datetime.datetime,
                 run_names: list[str],
                 processor_configs: list[ProcessorConfig]
    ) -> None:
        self._snapshot_id = snapshot_id
        self._create_time = create_time
        self._run_names = run_names
        self._processor_configs = processor_configs
    
    @property
    def snapshot_id(self) -> str:
        """The indentifier for this snapshot."""
        return self._snapshot_id
    
    @property
    def run_names(self) -> list[str]:
        """Alternate ids which may be used to identify this config snapshot."""
        return self._run_names
    
    @property
    def all_configs(self) -> list[ProcessorConfig]:
        """List of all configurations in this snapshot."""
        return self._processor_configs
    
    def get_config(self, name: str) -> ProcessorConfig | None:
        """Returns the configuration with the given name in this snapshot if it exists.

        Args:
            name: The name of the configuration.
        """
        for config in self._processor_configs:
            if name == config.name:
                return config
        
        return None

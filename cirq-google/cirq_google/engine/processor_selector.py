# Copyright 2023 The Cirq Developers
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

import attrs
from typing import Optional

from cirq_google.cloud import quantum
from cirq_google.engine.device_config_key import DeviceConfigKey


@attrs.frozen
class ProcessorSelector:
    """Selects a processor to run jobs.

    Args:
    processor_id: Processor identifier of the candidate to run the program.
    config_key: Unique identifier for the device configuration. If empty, it
        will use some internal default device configuration.
    """

    processor_id: str
    config_key: Optional[DeviceConfigKey] = None

    def to_quantum_processor_selector(
        self, project_id: str
    ) -> quantum.SchedulingConfig.ProcessorSelector:
        """Converts the processor selector into the Quantum Engine processor selector.

        Args:
        project_id: A project_id of the parent Google Cloud Project.
        """
        selector = quantum.SchedulingConfig.ProcessorSelector()
        selector.processor = self._processor_name_from_ids(project_id)
        if self.config_key:
            selector.device_config_key = self.config_key.to_quantum_device_config_key()
        return selector

    def _processor_name_from_ids(self, project_id: str) -> str:
        return f'projects/{project_id}/processors/{self.processor_id}'

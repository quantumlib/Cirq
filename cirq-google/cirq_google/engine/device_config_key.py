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

@attrs.frozen
class DeviceConfigKey:
    """Uniquely identifies a Device Configuration.

    Args:
    run_name: Identifier for the automation run. An Automation Run contains a
        collection of device configurations for a processor. If the value is None,
        it uses the internal default automation run.
    config_alias: Configuration alias used to identify a processor configuration
        within the automation run.
    """

    run_name: Optional[str]
    config_alias: str

    def to_quantum_device_config_key(self) -> quantum.DeviceConfigKey:
      """Converts the device configuration key into the Quantum Engine device configuration key."""
      return quantum.DeviceConfigKey(run_name=self.run_name, config_alias=self.config_alias)

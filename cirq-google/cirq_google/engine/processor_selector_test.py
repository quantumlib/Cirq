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

import pytest

from cirq_google.cloud import quantum
import cirq_google as cg


@pytest.mark.parametrize('project_id', ["project_id"])
@pytest.mark.parametrize('processor_id', ["processor_id"])
@pytest.mark.parametrize(
    'config_key',
    [None, cg.DeviceConfigKey("", "CONFIG_ALIAS"), cg.DeviceConfigKey("RUN_NAME", "CONFIG_ALIAS")],
)
def test_to_quantum_processor_selector(
    project_id: str, processor_id: str, config_key: cg.DeviceConfigKey
):
    selector = cg.ProcessorSelector(processor_id=processor_id, config_key=config_key)

    quantum_selector = selector.to_quantum_processor_selector(project_id=project_id)

    assert quantum_selector.processor == f'projects/{project_id}/processors/{processor_id}'
    assert not quantum_selector.processor_names
    if selector.config_key:
        assert (
            quantum_selector.device_config_key == selector.config_key.to_quantum_device_config_key()
        )
    else:
        assert quantum_selector.device_config_key == quantum.DeviceConfigKey()

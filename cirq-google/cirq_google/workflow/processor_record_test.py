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

import os
from unittest import mock

import cirq
import cirq_google as cg


def test_engine_backend():

    with mock.patch.dict(
        os.environ,
        {
            'GOOGLE_CLOUD_PROJECT': 'project!',
        },
        clear=True,
    ):
        processor = cg.EngineProcessorRecord('rainbow')
        assert processor.processor_id == 'rainbow'
        assert isinstance(processor.get_sampler(), cirq.Sampler)
    cirq.testing.assert_equivalent_repr(processor, global_vals={'cirq_google': cg})


def test_simulated_backend():
    with mock.patch.dict(
        os.environ,
        {
            'GOOGLE_CLOUD_PROJECT': 'project',
        },
        clear=True,
    ):
        processor = cg.SimulatedProcessorRecord('rainbow')
    assert processor.processor_id == 'rainbow'
    assert processor.descriptive_name() == 'rainbow-simulator'
    cirq.testing.assert_equivalent_repr(processor, global_vals={'cirq_google': cg})


def test_simulated_backend_with_local_device():
    processor = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    assert processor.processor_id == 'rainbow'
    assert processor.descriptive_name() == 'rainbow-simulator'

    cirq.testing.assert_equivalent_repr(processor, global_vals={'cirq_google': cg})

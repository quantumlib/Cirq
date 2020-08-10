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

import os
from unittest import mock
import warnings
import pytest

import cirq


@mock.patch('cirq.google.engine.client.quantum.QuantumEngineServiceClient')
def test_engine_from_environment(build):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Default project id present.
        env_dict = {'CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID': 'project!'}
        with mock.patch.dict(os.environ, env_dict, clear=True):
            eng = cirq.google.engine_from_environment()
            assert eng.project_id == 'project!'

        # Nothing present.
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError,
                               match='CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID'):
                _ = cirq.google.engine_from_environment()


def test_deprecation():
    with cirq.testing.assert_logs('engine_from_environment', 'get_engine',
                                  'deprecated'):
        env_dict = {'CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID': 'project!'}
        with mock.patch.dict(os.environ, env_dict, clear=True):
            _ = cirq.google.engine_from_environment()

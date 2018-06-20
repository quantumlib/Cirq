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
import pytest
from apiclient import discovery

import cirq
from cirq.testing.mock import mock


@mock.patch.dict(os.environ, {})
@mock.patch.object(discovery, 'build')
def test_engine_from_environment_nothing_set(build):
    with pytest.raises(EnvironmentError, match='not set'):
        eng = cirq.google.engine_from_environment()


@mock.patch.dict(os.environ, {
    'CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID': 'project!'
})
@mock.patch.object(discovery, 'build')
def test_engine_from_environment_missing_api_key(build):
    with pytest.raises(EnvironmentError,
                       match='CIRQ_QUANTUM_ENGINE_API_KEY'):
        _ = cirq.google.engine_from_environment()


@mock.patch.dict(os.environ, {'CIRQ_QUANTUM_ENGINE_API_KEY': 'key!'})
@mock.patch.object(discovery, 'build')
def test_engine_from_environment_missing_default_project_id(build):
    eng = cirq.google.engine_from_environment()
    assert eng.default_project_id is None
    assert eng.api_key == 'key!'


@mock.patch.dict(os.environ, {
    'CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID': 'project!',
    'CIRQ_QUANTUM_ENGINE_API_KEY': 'key!',
})
@mock.patch.object(discovery, 'build')
def test_engine_from_environment_all(build):
    eng = cirq.google.engine_from_environment()
    assert eng.default_project_id == 'project!'
    assert eng.api_key == 'key!'

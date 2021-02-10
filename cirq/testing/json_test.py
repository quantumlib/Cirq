# Copyright 2020 The Cirq Developers
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

from cirq.testing.json import spec_for


def test_module_missing_json_test_data():
    with pytest.raises(ValueError, match="json_test_data"):
        spec_for('cirq.testing.test_data.test_module_missing_json_test_data')


def test_module_missing_testspec():
    with pytest.raises(ValueError, match="TestSpec"):
        spec_for('cirq.testing.test_data.test_module_missing_testspec')


def test_missing_module():
    with pytest.raises(ModuleNotFoundError):
        spec_for('non_existent')

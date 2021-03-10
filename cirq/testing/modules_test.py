# Copyright 2021 The Cirq Developers
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

import cirq


def test_skip_module_decorators():
    f = lambda: 1
    assert cirq.testing.skip_if_module_not_exists(module="cirq")(f) == f
    assert cirq.testing.skip_if_module_exists(module="cirq")(f) is None

    # docs is not a valid python package but it is a directory
    assert cirq.testing.skip_if_module_not_exists(module="docs")(f) is None
    assert cirq.testing.skip_if_module_exists(module="docs")(f) == f

    assert cirq.testing.skip_if_module_not_exists(module="cirq.non_existent")(f) is None
    assert cirq.testing.skip_if_module_exists(module="cirq.non_existent")(f) == f

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

from __future__ import annotations

from dev_tools import modules


def test_versions_are_the_same() -> None:
    """Test for consistent version number across all modules."""
    mods = modules.list_modules(include_parent=True)
    versions = {m.name: m.version for m in mods}
    assert len(set(versions.values())) == 1, f"Versions should be the same, instead: \n{versions}"

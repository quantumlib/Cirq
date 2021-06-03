# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Any

from dev_tools import modules


def test_versions_are_the_same():
    mods = modules.list_modules(include_parent=True)
    versions = {m.name: m.version for m in mods}
    assert len(set(versions.values())) == 1, f"Versions should be the same, instead: \n{versions}"


def _get_version(package: str):
    version_file = f'{package}/_version.py'
    resulting_locals: Dict[str, Any] = {}
    exec(open(version_file).read(), globals(), resulting_locals)
    __version__ = resulting_locals['__version__']
    assert __version__, f"__version__ should be defined in {version_file}"
    return __version__

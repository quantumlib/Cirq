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
from codeowners import CodeOwners

CIRQ_MAINTAINERS = {('TEAM', "@quantumlib/cirq-maintainers"),
                    ('USERNAME', "@vtomole"), ('USERNAME', "@cduck")}


def _parse_owners():
    with open(".github/CODEOWNERS") as f:
        contents = f.read()
        return CodeOwners(contents)


owners = _parse_owners()


@pytest.mark.parametrize("pattern,expected",
                         [("any_file", CIRQ_MAINTAINERS),
                          ("in/any/dir/any_file.py", CIRQ_MAINTAINERS),
                          ("cirq/contrib/bla.py", CIRQ_MAINTAINERS)])
def test_codeowners(pattern, expected):
    assert set(owners.of(pattern)) == expected

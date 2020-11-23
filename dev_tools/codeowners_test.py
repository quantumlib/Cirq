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

CIRQ_MAINTAINERS = ('TEAM', "@quantumlib/cirq-maintainers")

BASE_MAINTAINERS = {
    CIRQ_MAINTAINERS, ('USERNAME', "@vtomole"), ('USERNAME', "@cduck")
}

GOOGLE_MAINTAINERS = {CIRQ_MAINTAINERS, ('USERNAME', "@wcourtney")}


@pytest.mark.parametrize("pattern,expected", [
    ("any_file", BASE_MAINTAINERS),
    ("in/any/dir/any_file.py", BASE_MAINTAINERS),
    ("cirq/contrib/bla.py", BASE_MAINTAINERS),
    ("cirq/google/test.py", GOOGLE_MAINTAINERS),
    ("cirq/google/in/any/dir/test.py", GOOGLE_MAINTAINERS),
    ("platforms/google/protos_as_well.proto", GOOGLE_MAINTAINERS),
    ("docs/google/notebook.ipynb", GOOGLE_MAINTAINERS),
    ("docs/tutorials/google/bla.md", GOOGLE_MAINTAINERS),
])
def test_codeowners(pattern, expected):
    # for some reason the codeowners library does not publish all the wheels
    # for Mac and Windows. Eventually we could write our own codeowners parser,
    # but for now it is good enough. If codeowners is not installed, this test
    # will be skipped
    try:
        from codeowners import CodeOwners
    except:
        pytest.skip("Skipping as codeowners not installed.")

    with open(".github/CODEOWNERS") as f:
        owners = CodeOwners(f.read())
        assert set(owners.of(pattern)) == expected

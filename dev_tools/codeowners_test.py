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

BASE_MAINTAINERS = {CIRQ_MAINTAINERS, ('USERNAME', "@vtomole"), ('USERNAME', "@cduck")}

GOOGLE_TEAM = {('USERNAME', "@wcourtney")}

GOOGLE_MAINTAINERS = BASE_MAINTAINERS.union(GOOGLE_TEAM)

IONQ_TEAM = {('USERNAME', u) for u in ["@dabacon", "@ColemanCollins", "@nakardo", "@gmauricio"]}
IONQ_MAINTAINERS = BASE_MAINTAINERS.union(IONQ_TEAM)

PASQAL_TEAM = {('USERNAME', u) for u in ["@HGSilveri"]}

PASQAL_MAINTAINERS = BASE_MAINTAINERS.union(PASQAL_TEAM)

AQT_TEAM = {('USERNAME', u) for u in ["@ma5x", "@pschindler", "@alfrisch"]}

AQT_MAINTAINERS = BASE_MAINTAINERS.union(AQT_TEAM)


def _vendor_module_testcases(mod_name, expected_group):
    return [
        (f"cirq/{mod_name}/test.py", expected_group),
        (f"cirq/{mod_name}/in/any/dir/test.py", expected_group),
        (f"platforms/{mod_name}/protos_as_well.proto", expected_group),
        (f"docs/{mod_name}/notebook.ipynb", expected_group),
        (f"docs/tutorials/{mod_name}/bla.md", expected_group),
    ]


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("any_file", BASE_MAINTAINERS),
        ("in/any/dir/any_file.py", BASE_MAINTAINERS),
        ("cirq/contrib/bla.py", BASE_MAINTAINERS),
        *_vendor_module_testcases("aqt", AQT_MAINTAINERS),
        *_vendor_module_testcases("ionq", IONQ_MAINTAINERS),
        *_vendor_module_testcases("google", GOOGLE_MAINTAINERS),
        *_vendor_module_testcases("pasqal", PASQAL_MAINTAINERS),
    ],
)
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

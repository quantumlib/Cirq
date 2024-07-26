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
import os

import pytest

CIRQ_MAINTAINERS = ('TEAM', "@quantumlib/cirq-maintainers")

BASE_MAINTAINERS = {CIRQ_MAINTAINERS, ('USERNAME', "@vtomole")}

DOCS_MAINTAINERS = BASE_MAINTAINERS.union({('USERNAME', '@aasfaw'), ('USERNAME', '@rmlarose')})

GOOGLE_TEAM = {('USERNAME', "@wcourtney"), ('USERNAME', "@verult")}

GOOGLE_MAINTAINERS = BASE_MAINTAINERS.union(GOOGLE_TEAM)

IONQ_TEAM = {
    ('USERNAME', u)
    for u in ["@dabacon", "@ColemanCollins", "@nakardo", "@gmauricio", "@Cynocracy", "@splch"]
}
IONQ_MAINTAINERS = BASE_MAINTAINERS.union(IONQ_TEAM)

PASQAL_TEAM = {('USERNAME', u) for u in ["@HGSilveri"]}

PASQAL_MAINTAINERS = BASE_MAINTAINERS.union(PASQAL_TEAM)

AQT_TEAM = {('USERNAME', u) for u in ["@ma5x", "@pschindler", "@alfrisch"]}

AQT_MAINTAINERS = BASE_MAINTAINERS.union(AQT_TEAM)

QCVV_TEAM = {('USERNAME', "@mrwojtek")}

QCVV_MAINTAINERS = BASE_MAINTAINERS.union(QCVV_TEAM)


@pytest.mark.parametrize(
    "filepath,expected",
    [
        ("setup.py", BASE_MAINTAINERS),
        ("dev_tools/codeowners_test.py", BASE_MAINTAINERS),
        ("cirq-core/setup.py", BASE_MAINTAINERS),
        ("cirq-core/cirq/contrib/__init__.py", BASE_MAINTAINERS),
        ("docs/_book.yaml", DOCS_MAINTAINERS),
        # qcvv
        ("cirq-core/cirq/experiments/__init__.py", QCVV_MAINTAINERS),
        ("docs/noise/qcvv/isolated_xeb.ipynb", QCVV_MAINTAINERS.union(DOCS_MAINTAINERS)),
        # aqt
        ("cirq-aqt/cirq_aqt/__init__.py", AQT_MAINTAINERS),
        ("cirq-aqt/setup.py", AQT_MAINTAINERS),
        ("docs/hardware/aqt/access.md", AQT_MAINTAINERS.union(DOCS_MAINTAINERS)),
        ("docs/hardware/aqt/getting_started.ipynb", AQT_MAINTAINERS.union(DOCS_MAINTAINERS)),
        # pasqal
        ("cirq-pasqal/cirq_pasqal/__init__.py", PASQAL_MAINTAINERS),
        ("cirq-pasqal/setup.py", PASQAL_MAINTAINERS),
        ("docs/hardware/pasqal/access.md", PASQAL_MAINTAINERS.union(DOCS_MAINTAINERS)),
        ("docs/hardware/pasqal/getting_started.ipynb", PASQAL_MAINTAINERS.union(DOCS_MAINTAINERS)),
        # ionq
        ("cirq-ionq/cirq_ionq/__init__.py", IONQ_MAINTAINERS),
        ("cirq-ionq/setup.py", IONQ_MAINTAINERS),
        ("docs/hardware/ionq/access.md", IONQ_MAINTAINERS.union(DOCS_MAINTAINERS)),
        ("docs/hardware/ionq/getting_started.ipynb", IONQ_MAINTAINERS.union(DOCS_MAINTAINERS)),
        # google
        ("cirq-google/cirq_google/__init__.py", GOOGLE_MAINTAINERS),
        ("cirq-google/setup.py", GOOGLE_MAINTAINERS),
        ("docs/google/access.md", GOOGLE_MAINTAINERS.union(DOCS_MAINTAINERS)),
        ("docs/tutorials/google/start.ipynb", GOOGLE_MAINTAINERS.union(DOCS_MAINTAINERS)),
    ],
)
def test_codeowners(filepath, expected):
    # for some reason the codeowners library does not publish all the wheels
    # for Mac and Windows. Eventually we could write our own codeowners parser,
    # but for now it is good enough. If codeowners is not installed, this test
    # will be skipped
    codeowners = pytest.importorskip("codeowners")

    with open(".github/CODEOWNERS") as f:
        owners = codeowners.CodeOwners(f.read())
        assert os.path.exists(
            filepath
        ), f"{filepath} should exist to avoid creating/maintaining meaningless codeowners rules."
        assert set(owners.of(filepath)) == expected

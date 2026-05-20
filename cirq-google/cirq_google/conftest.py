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

import os
from unittest import mock

import matplotlib.pyplot as plt
import pytest


def pytest_configure(config):
    os.environ['CIRQ_TESTING'] = "true"


@pytest.fixture
def closefigures():
    yield
    plt.close('all')


@pytest.fixture(scope="session", autouse=True)
def disable_local_gcloud_credentials(tmp_path_factory):
    """Ensure tests cannot authenticate to cloud servers with user credentials.

    Disable credentials set either through environment variable GOOGLE_APPLICATION_CREDENTIALS
    or via configuration files if user ran `gcloud auth application-default login`

    For more details see `google.auth.default`.
    """
    empty_dir = tmp_path_factory.mktemp("empty_gcloud_config-cirq_google", numbered=False)
    scrubbed_environ = {
        **{k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"},
        "CLOUDSDK_CONFIG": str(empty_dir),
    }
    with mock.patch.dict(os.environ, scrubbed_environ, clear=True):
        yield

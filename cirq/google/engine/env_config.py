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

"""Utility methods for getting configured Engine instances."""

import os

from cirq.google.engine.engine import Engine

ENV_PROJECT_ID = 'CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID'


def engine_from_environment() -> Engine:
    """Returns an Engine instance configured using environment variables.

    If the environment variables are set, but incorrect, an authentication
    failure will occur when attempting to run jobs on the engine.

    Required Environment Variables:
        QUANTUM_ENGINE_PROJECT: The name of a google cloud project, with the
            quantum engine enabled, that you have access to.

    Raises:
        EnvironmentError: The environment variables are not set.
    """
    project_id = os.environ.get(ENV_PROJECT_ID)
    if not project_id:
        raise EnvironmentError(
            'Environment variable {} is not set.'.format(ENV_PROJECT_ID))

    return Engine(project_id=project_id)

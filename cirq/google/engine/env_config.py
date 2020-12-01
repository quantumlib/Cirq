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

from typing import TYPE_CHECKING

from cirq.google import engine
from cirq._compat import deprecated

if TYPE_CHECKING:
    import cirq

ENV_PROJECT_ID = 'CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID'


@deprecated(deadline='v0.10.0', fix='Use cirq.get_engine instead.')
def engine_from_environment() -> 'cirq.google.Engine':
    """Returns an Engine instance configured using environment variables.

    If the environment variables are set, but incorrect, an authentication
    failure will occur when attempting to run jobs on the engine.

    Required Environment Variables:
        CIRQ_QUANTUM_ENGINE_DEFAULT_PROJECT_ID: The name of a google cloud
            project, with the quantum engine enabled, that you have access to.

    Raises:
        EnvironmentError: The environment variables are not set.
    """
    project_id = os.environ.get(ENV_PROJECT_ID)
    if not project_id:
        raise EnvironmentError('Environment variable {} is not set.'.format(ENV_PROJECT_ID))
    return engine.get_engine(project_id)

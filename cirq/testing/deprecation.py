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
import re
from contextlib import contextmanager

from cirq.testing import assert_logs

ALLOW_DEPRECATION_IN_TEST = 'ALLOW_DEPRECATION_IN_TEST'

DEADLINE_REGEX=r"v(\d)+\.(\d)+.(\d)+"


@contextmanager
def assert_deprecated(*msgs: str, deadline: str):
    """Allows deprecated functions, classes, decorators in tests.

    It acts as a contextmanager that can be used in with statements:
    >>> with assert_deprecated("use cirq.x instead", deadline="v0.9"):
    >>>     # do something deprecated
    """

    assert re.match(DEADLINE_REGEX, deadline), "deadline should match vX.Y.Z"

    os.environ[ALLOW_DEPRECATION_IN_TEST] = 'True'
    try:
        with assert_logs(*(msgs + (deadline,))):
            yield True
    finally:
        del os.environ[ALLOW_DEPRECATION_IN_TEST]

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
from contextlib import contextmanager

from cirq.testing import assert_logs

ALLOW_DEPRECATION_IN_TEST = 'ALLOW_DEPRECATION_IN_TEST'


@contextmanager
def assert_deprecated(*msgs: str, deadline: str, allow_multiple_warnings: bool = False):
    """Allows deprecated functions, classes, decorators in tests.

    It acts as a contextmanager that can be used in with statements:
    >>> with assert_deprecated("use cirq.x instead", deadline="v0.9"):
    >>>     # do something deprecated

    Args:
        msgs: messages that should match the warnings captured
        deadline: the expected deadline the feature will be deprecated by. Has to follow the format
            vX.Y (minor versions only)
        allow_multiple_warnings: if True, multiple warnings are accepted. Typically this should not
            be used, by default it's False.
    """

    os.environ[ALLOW_DEPRECATION_IN_TEST] = 'True'
    try:
        with assert_logs(*(msgs + (deadline,)), count=None if allow_multiple_warnings else 1):
            yield True
    finally:
        del os.environ[ALLOW_DEPRECATION_IN_TEST]

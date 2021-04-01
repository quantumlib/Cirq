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
import logging
import os
from contextlib import contextmanager
from typing import Optional

from cirq._compat import deprecated_parameter
from cirq.testing import assert_logs

ALLOW_DEPRECATION_IN_TEST = 'ALLOW_DEPRECATION_IN_TEST'


@contextmanager
@deprecated_parameter(
    deadline='v0.12',
    fix='Use count instead.',
    parameter_desc='allow_multiple_warnings',
    match=lambda args, kwargs: 'allow_multiple_warnings' in kwargs,
    rewrite=lambda args, kwargs: (
        args,
        dict(
            ('count', None if v == True else 1) if k == 'allow_multiple_warnings' else (k, v)
            for k, v in kwargs.items()
        ),
    ),
)
def assert_deprecated(*msgs: str, deadline: str, count: Optional[int] = 1):
    """Allows deprecated functions, classes, decorators in tests.

    It acts as a contextmanager that can be used in with statements:
    >>> with assert_deprecated("use cirq.x instead", deadline="v0.9"):
    >>>     # do something deprecated

    Args:
        msgs: messages that should match the warnings captured
        deadline: the expected deadline the feature will be deprecated by. Has to follow the format
            vX.Y (minor versions only)
        count: if None count of messages is not asserted, otherwise the number of deprecation
            messages have to equal count.
    """

    orig_exist, orig_value = (
        ALLOW_DEPRECATION_IN_TEST in os.environ,
        os.environ.get(ALLOW_DEPRECATION_IN_TEST, None),
    )
    os.environ[ALLOW_DEPRECATION_IN_TEST] = 'True'
    try:
        with assert_logs(
            *(msgs + (deadline,)),
            min_level=logging.WARNING,
            max_level=logging.WARNING,
            count=count,
        ):
            yield True
    finally:
        try:
            if orig_exist:
                # mypy can't resolve that orig_exist ensures that orig_value
                # of type Optional[str] can't be None
                os.environ[ALLOW_DEPRECATION_IN_TEST] = orig_value  # type: ignore
            else:
                del os.environ[ALLOW_DEPRECATION_IN_TEST]
        except:
            # this is only for nested deprecation checks
            pass

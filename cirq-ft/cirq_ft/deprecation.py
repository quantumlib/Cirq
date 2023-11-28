# Copyright 2023 The Cirq Developers
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
from typing import Callable, Type
from cirq._compat import deprecated, deprecated_class
from cirq.testing.deprecation import assert_deprecated
import functools

DEPRECATION_DEADLINE = 'v1.4'
_DEPRECATION_FIX_MSG = "Cirq-FT is deprecated in favour of Qualtran. pip install qualtran instead."


def deprecated_cirq_ft_class() -> Callable[[Type], Type]:
    return deprecated_class(deadline=DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)


def deprecated_cirq_ft_function() -> Callable[[Callable], Callable]:
    return deprecated(deadline=DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)


def call_with_assert_deprecated(obj, *args, **kwargs):
    with assert_deprecated(deadline=DEPRECATION_DEADLINE, count=None):
        return obj(*args, **kwargs)


def allow_deprecated_cirq_ft_use_in_tests(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import os
        from cirq.testing.deprecation import ALLOW_DEPRECATION_IN_TEST
        from cirq.testing import assert_logs
        import logging

        orig_exist, orig_value = (
            ALLOW_DEPRECATION_IN_TEST in os.environ,
            os.environ.get(ALLOW_DEPRECATION_IN_TEST, None),
        )

        os.environ[ALLOW_DEPRECATION_IN_TEST] = 'True'
        try:
            with assert_logs(
                min_level=logging.WARNING, max_level=logging.WARNING, count=None
            ) as logs:
                ret_val = func(*args, **kwargs)
            for log in logs:
                msg = log.getMessage()
                if _DEPRECATION_FIX_MSG in msg:
                    assert DEPRECATION_DEADLINE in msg
            return ret_val
        finally:
            if orig_exist:
                # mypy can't resolve that orig_exist ensures that orig_value
                # of type Optional[str] can't be None
                os.environ[ALLOW_DEPRECATION_IN_TEST] = orig_value  # pragma: no cover
            else:
                del os.environ[ALLOW_DEPRECATION_IN_TEST]

    return wrapper

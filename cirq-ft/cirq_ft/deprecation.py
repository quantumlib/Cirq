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
import functools
import unittest.mock
import os
from typing import Callable, Type
from cirq._compat import deprecated, deprecated_class

_DEPRECATION_DEADLINE = 'v1.4'
_DEPRECATION_FIX_MSG = "Cirq-FT is deprecated in favour of Qualtran. pip install qualtran instead."


def deprecated_cirq_ft_class() -> Callable[[Type], Type]:  # coverage: ignore
    """Decorator to mark a class in Cirq-FT deprecated."""
    return deprecated_class(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)


def deprecated_cirq_ft_function() -> Callable[[Callable], Callable]:  # coverage: ignore
    """Decorator to mark a function in Cirq-FT deprecated."""
    return deprecated(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)


def allow_deprecated_cirq_ft_use_in_tests(func):  # coverage: ignore
    """Decorator to allow using deprecated classes and functions in Tests and suppress warnings."""

    @functools.wraps(func)
    @unittest.mock.patch.dict(os.environ, ALLOW_DEPRECATION_IN_TEST="True")
    def wrapper(*args, **kwargs):
        from cirq.testing import assert_logs
        import logging

        with assert_logs(min_level=logging.WARNING, max_level=logging.WARNING, count=None) as logs:
            ret_val = func(*args, **kwargs)
        for log in logs:
            msg = log.getMessage()
            if _DEPRECATION_FIX_MSG in msg:
                assert _DEPRECATION_DEADLINE in msg
        return ret_val

    return wrapper

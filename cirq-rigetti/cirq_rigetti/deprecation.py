# Copyright 2025 The Cirq Developers
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
import os
import unittest.mock
from typing import Callable, Type

from cirq._compat import block_overlapping_deprecation, deprecated, deprecated_class

_DEPRECATION_DEADLINE = "v1.6"
_DEPRECATION_FIX_MSG = (
    "Cirq-Rigetti is deprecated.  For more details or to provide feedback see "
    "https://github.com/quantumlib/Cirq/issues/7058"
)


def deprecated_cirq_rigetti_class() -> Callable[[Type], Type]:  # coverage: ignore
    """Decorator to mark a class in Cirq-Rigetti deprecated."""
    return deprecated_class(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)


def deprecated_cirq_rigetti_function() -> Callable[[Callable], Callable]:  # coverage: ignore
    """Decorator to mark a function in Cirq-Rigetti deprecated."""
    return deprecated(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)


def allow_deprecated_cirq_rigetti_use_in_tests(func):  # coverage: ignore
    """Decorator to allow deprecated classes and functions in tests and to suppress warnings."""

    @functools.wraps(func)
    @unittest.mock.patch.dict(os.environ, ALLOW_DEPRECATION_IN_TEST="True")
    def wrapper(*args, **kwargs):
        with block_overlapping_deprecation(_DEPRECATION_FIX_MSG):
            return func(*args, **kwargs)

    return wrapper

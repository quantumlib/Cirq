# Copyright 2026 The Cirq Developers
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
from collections.abc import Callable

from cirq._compat import deprecated, deprecated_class
from cirq.testing.deprecation import assert_deprecated

_DEPRECATION_DEADLINE = "v1.9"
_DEPRECATION_FIX_MSG = (
    "cirq-web is deprecated.  For more details or to provide feedback see "
    "https://github.com/quantumlib/Cirq/issues/8168"
)


def deprecated_cirq_web_class(cls: type) -> Callable[[type], type]:
    """Decorator to mark a class in cirq-web deprecated."""
    return deprecated_class(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)(cls)


def deprecated_cirq_web_function(func: Callable) -> Callable:
    """Decorator to mark a function in cirq-web deprecated."""
    return deprecated(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)(func)


def assert_deprecated_cirq_web_warning(func: Callable) -> Callable:
    """Decorator to allow deprecated cirq-web code in tests and to verify it emits warnings."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with assert_deprecated(_DEPRECATION_FIX_MSG, deadline=_DEPRECATION_DEADLINE, count=None):
            return func(*args, **kwargs)

    return wrapper

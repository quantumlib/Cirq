# Copyright 2024 The Cirq Developers
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

"""Support one retry of tests that fail for a specific seed from pytest-randomly."""

import functools
import warnings
from typing import Any, Callable


def retry_once_with_later_random_values(testfunc: Callable) -> Callable:
    """Marks a test function for one retry with later random values.

    This decorator is intended for test functions which occasionally fail
    for specific random seeds from pytest-randomly.
    """

    @functools.wraps(testfunc)
    def wrapped_func(*args, **kwargs) -> Any:
        try:
            return testfunc(*args, **kwargs)
        except AssertionError:
            pass
        warnings.warn("Retrying in case we got a failing seed from pytest-randomly.")
        return testfunc(*args, **kwargs)

    return wrapped_func

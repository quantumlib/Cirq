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
import warnings

import pytest

from cirq.testing import assert_deprecated


def test_nested_assert_deprecation():
    with assert_deprecated(deadline="v1.2", count=1):
        with assert_deprecated(deadline="v1.2", count=1):
            with assert_deprecated(deadline="v1.2", count=1):
                warnings.warn("hello, this is deprecated in v1.2")


def test_assert_deprecated_log_handling():
    # correct deprecation message
    with assert_deprecated("hello", deadline="v1.2"):
        warnings.warn("hello, this is deprecated in v1.2")

    # missed deprecation warning
    with pytest.raises(AssertionError, match="Expected 1 log message but got 0."):
        with assert_deprecated(deadline="v1.2"):
            pass

    # too many deprecation warnings (only 1 should be emitted!)
    with pytest.raises(AssertionError, match="Expected 1 log message but got 2."):
        with assert_deprecated(deadline="v1.2"):
            warnings.warn("hello, this is deprecated in v1.2")
            warnings.warn("hello, this is deprecated in v1.2")

    # allowing for multiple deprecation warnings (in case of json serialization for multiple objects
    # for example)
    with assert_deprecated(deadline="v1.2", count=2):
        warnings.warn("hello, this is deprecated in v1.2")
        warnings.warn("hello, this is deprecated in v1.2")

    with assert_deprecated(deadline="v1.2", count=None):
        warnings.warn("hello, this is deprecated in v1.2")
        warnings.warn("hello, this is deprecated in v1.2")
        warnings.warn("hello, this is deprecated in v1.2")


def test_deprecated():
    # allow_multiple_warnings is now deprecated...so this is a bit convoluted,
    # a parameter of the deprecator is being deprecated

    with assert_deprecated(deadline="v0.12", count=3):
        with pytest.raises(AssertionError, match="Expected 1 log message but got 2."):
            # pylint: disable=unexpected-keyword-arg
            with assert_deprecated(deadline="v1.2", allow_multiple_warnings=False):
                warnings.warn("hello, this is deprecated in v1.2")
                warnings.warn("hello, this is deprecated in v1.2")

    with assert_deprecated(deadline="v0.12", count=3):
        # pylint: disable=unexpected-keyword-arg
        with assert_deprecated(deadline="v1.2", allow_multiple_warnings=True):
            warnings.warn("hello, this is deprecated in v1.2")
            warnings.warn("hello, this is deprecated in v1.2")

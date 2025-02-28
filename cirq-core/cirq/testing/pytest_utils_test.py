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

from unittest.mock import Mock

import pytest

import cirq


def test_retry_once_after_timeout():
    testfunc = Mock(side_effect=[TimeoutError("first call fails"), None])
    decoratedfunc = cirq.testing.retry_once_after_timeout(testfunc)
    with pytest.warns(UserWarning, match="Retrying.*transitive TimeoutError"):
        decoratedfunc()
    assert testfunc.call_count == 2


def test_retry_once_with_later_random_values():
    testfunc = Mock(side_effect=[AssertionError("first call fails"), None])
    decoratedfunc = cirq.testing.retry_once_with_later_random_values(testfunc)
    with pytest.warns(UserWarning, match="Retrying.*failing seed.*pytest-randomly"):
        decoratedfunc()
    assert testfunc.call_count == 2

# Copyright 2019 The Cirq Developers
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

import asyncio

import pytest

import cirq


def test_assert_still_running():
    f = asyncio.Future()
    cirq.testing.assert_asyncio_still_running(f)

    f.set_result(5)
    with pytest.raises(AssertionError, match="Not running"):
        cirq.testing.assert_asyncio_still_running(f)

    e = asyncio.Future()
    e.set_exception(ValueError('test fail'))
    with pytest.raises(ValueError, match="test fail"):
        cirq.testing.assert_asyncio_still_running(e)


def test_assert_will_have_result():
    f = asyncio.Future()
    with pytest.raises(AssertionError, match="Not done"):
        cirq.testing.assert_asyncio_will_have_result(f, 5, timeout=0.01)

    f.set_result(5)
    assert cirq.testing.assert_asyncio_will_have_result(f, 5) == 5
    with pytest.raises(AssertionError, match="!="):
        cirq.testing.assert_asyncio_will_have_result(f, 6)

    e = asyncio.Future()
    e.set_exception(ValueError('test fail'))
    with pytest.raises(ValueError, match="test fail"):
        cirq.testing.assert_asyncio_will_have_result(e, 5)


def test_assert_will_raise():
    f = asyncio.Future()
    with pytest.raises(AssertionError, match="Not done"):
        cirq.testing.assert_asyncio_will_raise(f,
                                               ValueError,
                                               match='',
                                               timeout=0.01)

    f.set_result(5)
    with pytest.raises(BaseException, match="DID NOT RAISE"):
        cirq.testing.assert_asyncio_will_raise(f, ValueError, match='')

    e = asyncio.Future()
    e.set_exception(ValueError('test fail'))
    cirq.testing.assert_asyncio_will_raise(e, ValueError, match="test fail")

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

from cirq.work.work_pool import CompletionOrderedAsyncWorkPool


def assert_still_running(future):
    try:
        asyncio.get_event_loop().run_until_complete(
            asyncio.wait_for(asyncio.shield(future), timeout=0.001)
        )
        assert False, "Not running: {!r}".format(future)
    except asyncio.TimeoutError:
        pass


def assert_will_have_result(future, expected):
    try:
        actual = asyncio.get_event_loop().run_until_complete(
            asyncio.wait_for(asyncio.shield(future), timeout=0.1))
        assert actual == expected, "<{!r}> != <{!r}> from <{!r}>".format(
            actual, expected, future)
    except asyncio.TimeoutError:
        assert False, "Not done: {}".format(future)


def assert_will_raise(future, expected, *, match):
    try:
        with pytest.raises(expected, match=match):
            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(asyncio.shield(future), timeout=0.1))
    except asyncio.TimeoutError:
        assert False, "Not done: {}".format(future)


def test_assert_still_running():
    f = asyncio.Future()
    assert_still_running(f)

    f.set_result(5)
    with pytest.raises(AssertionError, match="Not running"):
        assert_still_running(f)

    e = asyncio.Future()
    e.set_exception(ValueError('test fail'))
    with pytest.raises(ValueError, match="test fail"):
        assert_still_running(e)


def test_assert_will_have_result():
    f = asyncio.Future()
    with pytest.raises(AssertionError, match="Not done"):
        assert_will_have_result(f, 5)

    f.set_result(5)
    assert_will_have_result(f, 5)
    with pytest.raises(AssertionError, match="!="):
        assert_will_have_result(f, 6)

    e = asyncio.Future()
    e.set_exception(ValueError('test fail'))
    with pytest.raises(ValueError, match="test fail"):
        assert_will_have_result(e, 5)


def test_assert_will_raise():
    f = asyncio.Future()
    with pytest.raises(AssertionError, match="Not done"):
        assert_will_raise(f, ValueError, match='')

    f.set_result(5)
    with pytest.raises(BaseException, match="DID NOT RAISE"):
        assert_will_raise(f, ValueError, match='')

    e = asyncio.Future()
    e.set_exception(ValueError('test fail'))
    assert_will_raise(e, ValueError, match="test fail")


def test_awaitable_future():
    f = AwaitableFuture()
    assert_still_running(f)
    assert_still_running(f)
    f.set_result(5)
    assert_will_have_result(f, 5)


def test_enqueue_then_dequeue():
    pool = CompletionOrderedAsyncWorkPool()
    work = asyncio.Future()
    pool.include_work(work)
    result = pool.__anext__()
    assert_still_running(result)
    work.set_result(5)
    assert_will_have_result(result, 5)


def test_dequeue_then_enqueue():
    pool = CompletionOrderedAsyncWorkPool()
    work = asyncio.Future()
    result = pool.__anext__()
    pool.include_work(work)
    assert_still_running(result)
    work.set_result(5)
    assert_will_have_result(result, 5)


def test_dequeue_then_done():
    pool = CompletionOrderedAsyncWorkPool()
    result = pool.__anext__()
    assert_still_running(result)
    pool.set_all_work_received_flag()
    assert_will_raise(result, StopAsyncIteration, match='no_more_work')


def test_done_then_dequeue():
    pool = CompletionOrderedAsyncWorkPool()
    pool.set_all_work_received_flag()
    result = pool.__anext__()
    assert_will_raise(result, StopAsyncIteration, match='no_more_work')


def test_ordering():
    pool = CompletionOrderedAsyncWorkPool()
    w1 = asyncio.Future()
    w2 = asyncio.Future()
    w3 = asyncio.Future()
    pool.include_work(w1)
    pool.include_work(w2)
    r1 = pool.__anext__()
    pool.include_work(w3)
    r2 = pool.__anext__()
    r3 = pool.__anext__()

    assert_still_running(r1)
    w2.set_result(6)
    assert_will_have_result(r1, 6)

    assert_still_running(r2)
    w1.set_result(7)
    assert_will_have_result(r2, 7)

    assert_still_running(r3)
    w3.set_result(8)
    assert_will_have_result(r3, 8)



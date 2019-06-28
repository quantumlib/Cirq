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

import cirq
from cirq.work.work_pool import CompletionOrderedAsyncWorkPool


def test_enqueue_then_dequeue():
    pool = CompletionOrderedAsyncWorkPool()
    work = asyncio.Future()
    pool.include_work(work)
    result = pool.__anext__()
    cirq.testing.assert_asyncio_still_running(result)
    work.set_result(5)
    cirq.testing.assert_asyncio_will_have_result(result, 5)


def test_async_all_done_pre_flag():
    pool = CompletionOrderedAsyncWorkPool()
    done = pool.async_all_done()

    cirq.testing.assert_asyncio_still_running(done)
    pool.set_all_work_received_flag()
    cirq.testing.assert_asyncio_will_have_result(done, None)


def test_async_all_done_post_flag():
    pool = CompletionOrderedAsyncWorkPool()
    pool.set_all_work_received_flag()
    cirq.testing.assert_asyncio_will_have_result(pool.async_all_done(), None)
    cirq.testing.assert_asyncio_will_have_result(pool.async_all_done(), None)


def test_async_all_done_finish_work_then_flag():
    pool = CompletionOrderedAsyncWorkPool()
    done = pool.async_all_done()

    work = asyncio.Future()
    pool.include_work(work)
    work.set_result(5)

    cirq.testing.assert_asyncio_still_running(done)
    pool.set_all_work_received_flag()
    cirq.testing.assert_asyncio_will_have_result(done, None)


def test_async_all_done_flag_then_finish_work():
    pool = CompletionOrderedAsyncWorkPool()
    done = pool.async_all_done()

    work = asyncio.Future()
    pool.include_work(work)
    pool.set_all_work_received_flag()

    cirq.testing.assert_asyncio_still_running(done)
    work.set_result(5)
    cirq.testing.assert_asyncio_will_have_result(done, None)


def test_enqueue_then_dequeue_with_failure():
    pool = CompletionOrderedAsyncWorkPool()
    assert pool.num_active == 0
    work = asyncio.Future()
    pool.include_work(work)
    assert pool.num_active == 1
    result = pool.__anext__()
    cirq.testing.assert_asyncio_still_running(result)
    assert pool.num_active == 1
    work.set_exception(ValueError('test'))
    cirq.testing.assert_asyncio_will_raise(result, ValueError, match='test')
    assert pool.num_active == 0


def test_dequeue_then_enqueue():
    pool = CompletionOrderedAsyncWorkPool()
    work = asyncio.Future()
    result = pool.__anext__()
    pool.include_work(work)
    cirq.testing.assert_asyncio_still_running(result)
    work.set_result(5)
    cirq.testing.assert_asyncio_will_have_result(result, 5)


def test_dequeue_then_done():
    pool = CompletionOrderedAsyncWorkPool()
    result = pool.__anext__()
    cirq.testing.assert_asyncio_still_running(result)
    pool.set_all_work_received_flag()
    cirq.testing.assert_asyncio_will_raise(result,
                                           StopAsyncIteration,
                                           match='no_more_work')


def test_done_then_dequeue():
    pool = CompletionOrderedAsyncWorkPool()
    pool.set_all_work_received_flag()
    result = pool.__anext__()
    cirq.testing.assert_asyncio_will_raise(result,
                                           StopAsyncIteration,
                                           match='no_more_work')


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

    cirq.testing.assert_asyncio_still_running(r1)
    w2.set_result(6)
    cirq.testing.assert_asyncio_will_have_result(r1, 6)

    cirq.testing.assert_asyncio_still_running(r2)
    w1.set_result(7)
    cirq.testing.assert_asyncio_will_have_result(r2, 7)

    cirq.testing.assert_asyncio_still_running(r3)
    w3.set_result(8)
    cirq.testing.assert_asyncio_will_have_result(r3, 8)


def test_async_for():
    pool = CompletionOrderedAsyncWorkPool()

    async def consume():
        results = []
        async for result in pool:
            results.append(result)
        return results

    async def produce():
        w1 = asyncio.Future()
        w2 = asyncio.Future()
        w3 = asyncio.Future()
        pool.include_work(w1)
        pool.include_work(w2)
        w1.set_result(1)
        pool.include_work(w3)
        pool.set_all_work_received_flag()
        w2.set_result(2)
        w3.set_result(3)

    a = asyncio.ensure_future(produce())
    b = asyncio.ensure_future(consume())
    cirq.testing.assert_asyncio_will_have_result(asyncio.wait([a, b]))
    cirq.testing.assert_asyncio_will_have_result(a, None)
    cirq.testing.assert_asyncio_will_have_result(b, [1, 2, 3])

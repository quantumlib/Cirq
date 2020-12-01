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
from typing import Union, Awaitable, Coroutine


def asyncio_pending(
    future: Union[Awaitable, asyncio.Future, Coroutine], timeout: float = 0.001
) -> Awaitable[bool]:
    """Gives the given future a chance to complete, and determines if it didn't.

    This method is used in tests checking that a future actually depends on some
    given event having happened. The test can assert, before the event, that the
    future is still pending and then assert, after the event, that the future
    has a result.

    Args:
        future: The future that may or may not be able to resolve when given
            a bit of time.
        timeout: The number of seconds to wait for the future. This should
            generally be a small value (milliseconds) when expecting the future
            to not resolve, and a large value (seconds) when expecting the
            future to resolve.

    Returns:
        True if the future is still pending after the timeout elapses. False if
        the future did complete (or fail) or was already completed (or already
        failed).

    Examples:
        >>> import asyncio
        >>> import pytest
        >>> @pytest.mark.asyncio
        ... async def test_completion_only_when_expected():
        ...     f = asyncio.Future()
        ...     assert await cirq.testing.asyncio_pending(f)
        ...     f.set_result(5)
        ...     assert await f == 5
    """

    async def body():
        f = asyncio.shield(future)
        t = asyncio.ensure_future(asyncio.sleep(timeout))
        done, _ = await asyncio.wait([f, t], return_when=asyncio.FIRST_COMPLETED)
        t.cancel()
        return f not in done

    return _AwaitBeforeAssert(body())


class _AwaitBeforeAssert:
    def __init__(self, awaitable: Awaitable):
        self.awaitable = awaitable

    def __bool__(self):
        raise RuntimeError(
            'You forgot the "await" in "assert await cirq.testing.asyncio_pending(...)".'
        )

    def __await__(self):
        return self.awaitable.__await__()

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

import asyncio
import errno
import threading
from typing import Awaitable, Callable, Optional, TypeVar

import duet
from typing_extensions import ParamSpec

R = TypeVar('R')
P = ParamSpec("P")


class AsyncioExecutor:
    """Runs asyncio coroutines in a thread, exposes the results as duet futures.

    This lets us bridge between an asyncio event loop (which is what async grpc
    code uses) and duet (which is what cirq uses for asynchrony).
    """

    def __init__(self) -> None:
        loop_future: duet.AwaitableFuture[asyncio.AbstractEventLoop] = duet.AwaitableFuture()
        thread = threading.Thread(target=asyncio.run, args=(self._main(loop_future),), daemon=True)
        thread.start()
        self.loop = loop_future.result()

    @staticmethod
    async def _main(loop_future: duet.AwaitableFuture) -> None:
        def handle_exception(loop, context) -> None:  # pragma: no cover
            # Ignore PollerCompletionQueue errors (see https://github.com/grpc/grpc/issues/25364)
            exc = context.get("exception")
            if exc and isinstance(exc, BlockingIOError) and exc.errno == errno.EAGAIN:
                return
            loop.default_exception_handler(context)

        loop = asyncio.get_running_loop()
        loop.set_exception_handler(handle_exception)
        loop_future.set_result(loop)
        while True:
            await asyncio.sleep(1)

    def submit(
        self, func: Callable[P, Awaitable[R]], *args: P.args, **kwargs: P.kwargs
    ) -> duet.AwaitableFuture[R]:
        """Dispatch the given function to be run in an asyncio coroutine.

        Args:
            func: asyncio function which will be run in a separate thread.
                Will be called with *args and **kw and should return an asyncio
                awaitable.
            *args: Positional args to pass to func.
            **kwargs: Keyword args to pass to func.
        """
        future = asyncio.run_coroutine_threadsafe(func(*args, **kwargs), self.loop)
        return duet.AwaitableFuture.wrap(future)

    _instance: Optional['AsyncioExecutor'] = None

    @classmethod
    def instance(cls) -> 'AsyncioExecutor':
        """Returns a singleton AsyncioExecutor shared globally."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

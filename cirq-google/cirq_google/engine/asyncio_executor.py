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

from __future__ import annotations

import asyncio
import contextvars
import errno
import threading
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import ParamSpec, TYPE_CHECKING, TypeVar

import duet

if TYPE_CHECKING:
    import concurrent

R = TypeVar('R')
P = ParamSpec("P")


class AsyncioPreferredExecutor:
    """A view of the AsyncioExecutor that defaults to asyncio futures instead of duet futures."""

    def __init__(self, executor: AsyncioExecutor):
        self._executor = executor

    def submit(self, *args, **kwargs):
        current_override = self._executor._override_ctx.get()

        if current_override is None:
            with self._executor.asyncio_futures():
                return self._executor.submit(*args, **kwargs)
        else:
            return self._executor.submit(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._executor, name)


class AsyncioExecutor:
    """Runs asyncio coroutines in a thread, exposes the results as either duet or asyncio futures.

    When using duet futures (the default), this lets us bridge between an asyncio event loop (which
    is what async grpc code uses) and duet (which is what cirq uses for asynchrony).

    When using asyncio futures, this lets us make calls using a consistent gRPC channel and event
    loop to prevent awaiting futures bound to the wrong loop.
    """

    def __init__(self) -> None:
        loop_future: duet.AwaitableFuture[asyncio.AbstractEventLoop] = duet.AwaitableFuture()
        thread = threading.Thread(target=asyncio.run, args=(self._main(loop_future),), daemon=True)
        thread.start()
        self.loop = loop_future.result()
        # None (default = False) | True (Duet futures) | False (asyncio futures)
        self._override_ctx: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
            'use_duet_futures_override', default=None
        )

    @contextmanager
    def asyncio_futures(self):
        token = self._override_ctx.set(False)
        try:
            yield
        finally:
            self._override_ctx.reset(token)

    @contextmanager
    def duet_futures(self):
        token = self._override_ctx.set(True)
        try:
            yield
        finally:
            self._override_ctx.reset(token)

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
    ) -> Awaitable[R]:
        """Dispatch the given function to be run in an asyncio coroutine.

        Args:
            func: asyncio function which will be run in a separate thread.
                Will be called with *args and **kw and should return an asyncio
                awaitable.
            *args: Positional args to pass to func.
            **kwargs: Keyword args to pass to func.
        """
        future: concurrent.futures.Future = asyncio.run_coroutine_threadsafe(
            func(*args, **kwargs), self.loop  # type: ignore[arg-type]
        )
        if self._override_ctx.get() is not False:
            return duet.AwaitableFuture.wrap(future)
        else:
            return asyncio.wrap_future(future)

    _instance: AsyncioExecutor | None = None

    @classmethod
    def instance(cls) -> AsyncioExecutor:
        """Returns a singleton AsyncioExecutor shared globally."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

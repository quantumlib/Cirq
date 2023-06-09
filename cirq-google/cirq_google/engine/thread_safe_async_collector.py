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
from typing import Generic, TypeVar

import threading

import duet

T = TypeVar("T")


class _ThreadSafeAsyncCollector(duet.AsyncCollector, Generic[T]):
    """Thread-safe version of duet.AsyncCollector.

    The producer (caller of `.add`, `.done`, or `.error`) and consumer (caller which iterates over
    the collector) can be on different threads. There can be multiple producer threads but only a
    single consumer thread.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def add(self, value: T) -> None:
        with self._lock:
            super().add(value)

    def done(self) -> None:
        with self._lock:
            super().done()

    def error(self, error: Exception) -> None:
        with self._lock:
            super().error(error)

    async def __anext__(self) -> T:
        with self._lock:
            if not self._done and not self._buffer:
                self._waiter = duet.AwaitableFuture()
                self._lock.release()
                await self._waiter
                self._lock.acquire()
                self._waiter = None
            if self._buffer:
                return self._buffer.popleft()
            if self._error:
                raise self._error
            raise StopAsyncIteration()

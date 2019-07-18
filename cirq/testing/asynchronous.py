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
import re
from collections.abc import Awaitable, Coroutine
from typing import Any, Type, Union

# A placeholder default value used to detect that callers did not specify an
# 'expected' value argument in `assert_asyncio_will_have_result`, and so the
# result should be returned without checking it.
JUST_RETURN_RESULT = object()  # type: Any


def _run_loop_waiting_for(future: Union[Awaitable, asyncio.Future, Coroutine],
                          timeout: float):
    return asyncio.get_event_loop().run_until_complete(
        asyncio.wait_for(asyncio.shield(future), timeout=timeout))


def assert_asyncio_still_running(
        future: Union[Awaitable, asyncio.Future, Coroutine],
        timeout: float = 0.001):
    """Checks that the given asyncio future has not completed.

    Works by running the asyncio event loop for a short amount of time.

    Args:
        future: The future that should not yet be resolved.
        timeout: The number of seconds to wait for the future. Make sure this is
             a small value, because it holds up the passing test!

    Raises:
        AssertError: The future completed or failed within the timeout.
    """
    try:
        _run_loop_waiting_for(future, timeout)
        assert False, "Not running: {!r}".format(future)
    except asyncio.TimeoutError:
        pass


def assert_asyncio_will_have_result(
        future: Union[Awaitable, asyncio.Future, Coroutine],
        expected: Any = JUST_RETURN_RESULT,
        timeout: float = 1.0) -> Any:
    """Checks that the given asyncio future completes with the given value.

    Works by running the asyncio event loop for up to the given timeout.

    Args:
        future: The asyncio awaitable that should complete.
        expected: The result that the future should have after it completes.
            If not specified, nothing is asserted about the result.
        timeout: The maximum number of seconds to run the event loop until the
            future resolves.

    Returns:
        The future's result.

    Raises:
        AssertError: The future did not complete in time, or did not contain
            the expected result.
    """
    try:
        actual = _run_loop_waiting_for(future, timeout)
        if expected is not JUST_RETURN_RESULT:
            assert actual == expected, "{!r} != {!r} from {!r}".format(
                actual, expected, future)
        return actual
    except asyncio.TimeoutError:
        assert False, "Not done: {!r}".format(future)


def assert_asyncio_will_raise(
        future: Union[Awaitable, asyncio.Future, Coroutine],
        expected: Type,
        *,
        match: str,
        timeout: float = 1.0):
    """Checks that the given asyncio future fails with a matching error.

    Works by running the asyncio event loop for up to the given timeout.

    Args:
        future: The asyncio awaitable that should error.
        expected: The exception type that the future should end up containing.
        match: A regex that must match the exception's message.
        timeout: The maximum number of seconds to run the event loop until the
            future resolves.

    Raises:
        AssertError: The future did not resolve in time, or did not contain
            a matching exception.
    """
    try:
        _run_loop_waiting_for(future, timeout)
    except expected as exc:
        if match and not re.search(match, str(exc)):
            assert False, "Pattern '{}' not found in '{}'".format(match, exc)
    except asyncio.TimeoutError:
        assert False, "Not done: {!r}".format(future)
    else:
        assert False, "DID NOT RAISE {}".format(expected)

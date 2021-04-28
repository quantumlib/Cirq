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


@pytest.mark.asyncio
async def test_asyncio_pending():
    f = asyncio.Future()

    assert await cirq.testing.asyncio_pending(f)
    f.set_result(5)
    assert not await cirq.testing.asyncio_pending(f)
    assert not await cirq.testing.asyncio_pending(f, timeout=100)

    e = asyncio.Future()

    assert await cirq.testing.asyncio_pending(e)
    e.set_exception(ValueError('test fail'))
    assert not await cirq.testing.asyncio_pending(e)
    assert not await cirq.testing.asyncio_pending(e, timeout=100)


@pytest.mark.asyncio
async def test_asyncio_pending_common_mistake_caught():
    f = asyncio.Future()
    pending = cirq.testing.asyncio_pending(f)
    with pytest.raises(RuntimeError, match='forgot the "await"'):
        assert pending
    assert await pending

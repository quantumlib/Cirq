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
"""Tests for cirq.Sampler."""
import asyncio

import cirq


def test_async_sampler_fail():

    class FailingSampler(cirq.AsyncSampler):

        async def run_sweep_async(self, program, params, repetitions: int = 1):
            await asyncio.sleep(0.01)
            raise ValueError('test')

    cirq.testing.assert_asyncio_will_raise(FailingSampler().run_async(
        cirq.Circuit(), repetitions=1),
                                           ValueError,
                                           match='test')

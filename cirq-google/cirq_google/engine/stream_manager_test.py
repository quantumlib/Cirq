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

from typing import AsyncIterable, AsyncIterator
from unittest import mock

import duet
import asyncio

from cirq_google.cloud import quantum

from cirq_google.engine.stream_manager import StreamManager


def setup_fake_quantum_run_stream_client(client_constructor):
    grpc_client = _FakeQuantumRunStream()
    client_constructor.return_value = grpc_client
    return grpc_client


class _FakeQuantumRunStream:
    def __init__(self):
        self.request_count = 0

    def set_response_list(self, response_list):
        self._response_list = response_list

    async def quantum_run_stream(
        self, requests: AsyncIterator[quantum.QuantumRunStreamRequest] = None, **kwargs
    ) -> AsyncIterable[quantum.QuantumRunStreamResponse]:
        async def run_async_iterator():
            async for _ in requests:
                self.request_count += 1
                yield self._response_list.pop(0)

        await asyncio.sleep(0.0001)
        return run_async_iterator()


@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_create_job_and_get_results(client_constructor):
    fake_client = setup_fake_quantum_run_stream_client(client_constructor)

    manager = StreamManager(fake_client)
    expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    mock_responses = [quantum.QuantumRunStreamResponse(message_id='0', result=expected_result)]
    fake_client.set_response_list(mock_responses)

    async def run():
        return await manager.send(
            quantum.QuantumRunStreamRequest(
                message_id='0',
                parent='projects/proj',
                create_quantum_job=quantum.CreateQuantumJobRequest(
                    parent='projects/proj/programs/prog',
                    quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0'),
                ),
            )
        )

    actual_result = duet.sync(run)()

    # TODO(verult) test that response listener message IDs are being deleted

    assert actual_result == expected_result
    assert fake_client.request_count == 1

    # TODO(verult) debug test failure

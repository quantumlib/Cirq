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

import duet

from cirq_google.engine.stream_response_demux import _StreamResponseDemux
from cirq_google.engine.thread_safe_async_collector import _ThreadSafeAsyncCollector
from cirq_google.engine.asyncio_executor import AsyncioExecutor

from cirq_google.cloud import quantum


class StreamManager:
    """Manager for the QuantunRunStream stream.
    TODO(verult) expand
    """

    # TODO(verult) Does StreamResponseDemux need to be thread-safe?

    # TODO(verult) caller should make this manager a cached_property
    def __init__(self, grpc_client: quantum.QuantumEngineServiceAsyncClient) -> None:
        self._response_demux = _StreamResponseDemux(cancel_callback=self._cancel)
        self._request_queue = _ThreadSafeAsyncCollector()
        self._event_loop_running = False
        self._grpc_client = grpc_client

    def _cancel(self) -> None:
        pass  # TODO

    @property
    def _executor(self) -> AsyncioExecutor:
        # We must re-use a single Executor due to multi-threading issues in gRPC
        # clients: https://github.com/grpc/grpc/issues/25364.
        return AsyncioExecutor.instance()

    def send(
        self, request: quantum.QuantumRunStreamRequest
    ) -> duet.AwaitableFuture[quantum.QuantumRunStreamResponse]:
        """Sends a QuantumRunStreamRequest

        Returns:
            A future which is populated with the corresponding QuantumRunStreamResponse when it is
            available.
        """
        if not self._event_loop_running:
            self._event_loop_running = True
            self._executor.submit(self.event_loop)
            # TODO does the syntax of self.event_loop work here to pass in a function?

        response_future = self._response_demux.subscribe(request)
        self._request_queue.add(request)
        return response_future

    async def event_loop(self):
        """Main event loop running in the asyncio thread."""
        while True:
            responses = await self._grpc_client.quantum_run_stream(self._request_queue)
            async for resp in responses:
                self._response_demux.publish(resp)
        # TODO(#5996) error handling.

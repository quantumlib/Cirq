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

from typing import Dict
import asyncio

from google.api_core.exceptions import GoogleAPICallError

from cirq_google.cloud import quantum


class ResponseDemux:
    """A event demultiplexer for QuantumRunStreamResponses, as part of the async reactor pattern."""

    def __init__(self):
        self._subscribers: Dict[str, asyncio.Future] = {}

    def subscribe(self, message_id: str) -> asyncio.Future:
        """Subscribes to the QuantumRunStreamResponse with a matching ID as the

        Returns:
            A future for the response, to be fulfilled when publish is called.
        """
        response_future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._subscribers[message_id] = response_future
        return response_future

    def unsubscribe(self, message_id: str) -> None:
        """Indicates that the caller is no longer waiting for the response matching message_id."""
        if message_id in self._subscribers:
            del self._subscribers[message_id]

    def publish(self, response: quantum.QuantumRunStreamResponse) -> None:
        """Makes the response available to the subscriber with the matching message ID.

        The subscriber is unsubscribed afterwards.
        """
        if response.message_id not in self._subscribers:
            return

        future = self._subscribers.pop(response.message_id)
        if not future.done():
            future.set_result(response)

    def publish_exception(self, exception: GoogleAPICallError) -> None:
        """Publishes an exception to all outstanding futures."""
        for future in self._subscribers.values():
            if not future.done():
                future.set_exception(exception)
        self._subscribers = {}


class StreamManager:
    # TODO(verult) Implement
    pass

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
    """A event demultiplexer for QuantumRunStreamResponses, as part of the async reactor pattern.

    A caller can subscribe to the response matching a provided message ID. Only a single caller may
    subscribe to each ID.

    Another caller can subsequently publish a response. The future held by the subscriber with
    the matching message ID will then be fulfilled.

    A caller can also publish an exception to all subscribers.
    """

    def __init__(self):
        # [message ID] : [subscriber future]
        self._subscribers: Dict[str, asyncio.Future] = {}

    def subscribe(self, message_id: str) -> asyncio.Future:
        """Subscribes to the QuantumRunStreamResponse with a matching ID.

        There should only be one subscriber per message ID.

        Returns:
            A future for the response, to be fulfilled when publish is called.

        Raises:
            ValueError: when trying to subscribe to a message_id which already has a subscriber.
        """

        if message_id in self._subscribers:
            raise ValueError(f'There is already a subscriber for the message ID {message_id}')

        response_future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._subscribers[message_id] = response_future
        return response_future

    def publish(self, response: quantum.QuantumRunStreamResponse) -> None:
        """Makes the response available to the subscriber with the matching message ID.

        The subscriber is unsubscribed afterwards.

        If there are no subscribers waiting for the response, nothing happens.
        """
        future = self._subscribers.pop(response.message_id, None)
        if future and not future.done():
            future.set_result(response)

    def publish_exception(self, exception: GoogleAPICallError) -> None:
        """Publishes an exception to all outstanding futures."""
        for future in self._subscribers.values():
            if not future.done():
                future.set_exception(exception)
        self._subscribers.clear()


class StreamManager:
    # TODO(verult) Implement
    pass

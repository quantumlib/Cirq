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
import pytest
import google.api_core.exceptions as google_exceptions

from cirq_google.engine.stream_manager import ResponseDemux
from cirq_google.cloud import quantum

RESPONSE0 = quantum.QuantumRunStreamResponse(
    message_id='0', result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
)
RESPONSE1 = quantum.QuantumRunStreamResponse(
    message_id='1', result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
)
RESPONSE1_WITH_DIFFERENT_RESULT = quantum.QuantumRunStreamResponse(
    message_id='1', result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job2')
)


class TestResponseDemux:
    @pytest.fixture
    def demux(self) -> ResponseDemux:
        return ResponseDemux()

    @pytest.mark.asyncio
    async def test_one_subscribe_one_publish_subscriber_receives_response(self, demux):
        future = demux.subscribe(message_id='0')
        demux.publish(RESPONSE0)
        actual_response = await asyncio.wait_for(future, timeout=1)

        assert actual_response == RESPONSE0

    @pytest.mark.asyncio
    async def test_subscribe_twice_to_same_message_id_raises_error(self, demux):
        with pytest.raises(ValueError):
            demux.subscribe(message_id='0')
            demux.subscribe(message_id='0')

    @pytest.mark.asyncio
    async def test_out_of_order_response_publishes_to_subscribers_subscribers_receive_responses(
        self, demux
    ):
        future0 = demux.subscribe(message_id='0')
        future1 = demux.subscribe(message_id='1')
        demux.publish(RESPONSE1)
        demux.publish(RESPONSE0)
        actual_response0 = await asyncio.wait_for(future0, timeout=1)
        actual_response1 = await asyncio.wait_for(future1, timeout=1)

        assert actual_response0 == RESPONSE0
        assert actual_response1 == RESPONSE1

    @pytest.mark.asyncio
    async def test_message_id_does_not_exist_subscriber_never_receives_response(self, demux):
        future = demux.subscribe(message_id='0')
        demux.publish(RESPONSE1)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(future, timeout=1)

    @pytest.mark.asyncio
    async def test_no_subscribers_does_not_throw(self, demux):
        demux.publish(RESPONSE0)

        # expect no exceptions

    @pytest.mark.asyncio
    async def test_publishes_twice_for_same_message_id_future_unchanged(self, demux):
        future = demux.subscribe(message_id='1')
        demux.publish(RESPONSE1)
        demux.publish(RESPONSE1_WITH_DIFFERENT_RESULT)
        actual_response = await asyncio.wait_for(future, timeout=1)

        assert actual_response == RESPONSE1

    @pytest.mark.asyncio
    async def test_publish_exception_publishes_to_all_subscribers(self, demux):
        exception = google_exceptions.Aborted('aborted')
        future0 = demux.subscribe(message_id='0')
        future1 = demux.subscribe(message_id='1')
        demux.publish_exception(exception)

        with pytest.raises(google_exceptions.Aborted):
            await future0
        with pytest.raises(google_exceptions.Aborted):
            await future1

    @pytest.mark.asyncio
    async def test_publish_response_after_publishing_exception_does_not_change_futures(self, demux):
        exception = google_exceptions.Aborted('aborted')
        future0 = demux.subscribe(message_id='0')
        future1 = demux.subscribe(message_id='1')
        demux.publish_exception(exception)
        demux.publish(RESPONSE0)
        demux.publish(RESPONSE1)

        with pytest.raises(google_exceptions.Aborted):
            await future0
        with pytest.raises(google_exceptions.Aborted):
            await future1

    @pytest.mark.asyncio
    async def test_publish_exception_after_publishing_response_does_not_change_futures(self, demux):
        exception = google_exceptions.Aborted('aborted')
        future0 = demux.subscribe(message_id='0')
        future1 = demux.subscribe(message_id='1')
        demux.publish(RESPONSE0)
        demux.publish(RESPONSE1)
        demux.publish_exception(exception)
        actual_response0 = await asyncio.wait_for(future0, timeout=1)
        actual_response1 = await asyncio.wait_for(future1, timeout=1)

        assert actual_response0 == RESPONSE0
        assert actual_response1 == RESPONSE1

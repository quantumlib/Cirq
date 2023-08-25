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

from typing import AsyncIterable, AsyncIterator, Awaitable, List, Union
import asyncio
import concurrent
from unittest import mock

import duet
import pytest
import google.api_core.exceptions as google_exceptions

from cirq_google.engine.stream_manager import (
    _get_retry_request_or_raise,
    ProgramAlreadyExistsError,
    ResponseDemux,
    StreamError,
    StreamManager,
)
from cirq_google.cloud import quantum


Code = quantum.StreamError.Code


# ResponseDemux test suite constants
RESPONSE0 = quantum.QuantumRunStreamResponse(
    message_id='0', result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
)
RESPONSE1 = quantum.QuantumRunStreamResponse(
    message_id='1', result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
)
RESPONSE1_WITH_DIFFERENT_RESULT = quantum.QuantumRunStreamResponse(
    message_id='1', result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job2')
)


# StreamManager test suite constants
REQUEST_PROJECT_NAME = 'projects/proj'
REQUEST_PROGRAM = quantum.QuantumProgram(name='projects/proj/programs/prog')
REQUEST_JOB = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')


def setup_fake_quantum_run_stream_client(client_constructor, responses_and_exceptions):
    grpc_client = FakeQuantumRunStream(responses_and_exceptions)
    client_constructor.return_value = grpc_client
    return grpc_client


class FakeQuantumRunStream:
    """A fake Quantum Engine client which supports QuantumRunStream and CancelQuantumJob."""

    def __init__(
        self, responses_and_exceptions: List[Union[quantum.QuantumRunStreamResponse, BaseException]]
    ):
        self.stream_requests: List[quantum.QuantumRunStreamRequest] = []
        self.cancel_requests: List[quantum.CancelQuantumJobRequest] = []
        self.responses_and_exceptions = responses_and_exceptions

    async def quantum_run_stream(
        self, requests: AsyncIterator[quantum.QuantumRunStreamRequest], **kwargs
    ) -> Awaitable[AsyncIterable[quantum.QuantumRunStreamResponse]]:
        """Fakes the QuantumRunStream RPC.

        Expects the number of requests to be the same as len(self.responses_and_exceptions).

        For every request, a response or exception is popped from `self.responses_and_exceptions`.
        Before the next request:
            * If it is a response, it is sent back through the stream.
            * If it is an exception, the exception is raised.

        This fake does not support out-of-order responses.

        No responses are ever made if `self.responses_and_exceptions` is empty.
        """

        async def run_async_iterator():
            async for request in requests:
                self.stream_requests.append(request)

                if not self.responses_and_exceptions:
                    while True:
                        await asyncio.sleep(1)

                response_or_exception = self.responses_and_exceptions.pop(0)
                if isinstance(response_or_exception, BaseException):
                    raise response_or_exception
                response_or_exception.message_id = request.message_id
                yield response_or_exception

        await asyncio.sleep(0)
        return run_async_iterator()

    async def cancel_quantum_job(self, request: quantum.CancelQuantumJobRequest) -> None:
        self.cancel_requests.append(request)
        await asyncio.sleep(0)


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


class TestStreamManager:
    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_expects_result_response(self, client_constructor):
        async def test():
            async with duet.timeout_scope(5):
                # Arrange
                expected_result = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job0'
                )
                mock_responses = [quantum.QuantumRunStreamResponse(result=expected_result)]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses
                )
                manager = StreamManager(fake_client)

                # Act
                actual_result = await manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                manager.stop()

                # Assert
                assert actual_result == expected_result
                assert len(fake_client.stream_requests) == 1
                # assert that the first request is a CreateQuantumProgramAndJobRequest.
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_without_name_raises(self, client_constructor):
        async def test():
            async with duet.timeout_scope(5):
                # Arrange
                expected_result = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job0'
                )
                mock_responses = [quantum.QuantumRunStreamResponse(result=expected_result)]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses
                )
                manager = StreamManager(fake_client)

                with pytest.raises(ValueError, match='Program name must be set'):
                    await manager.submit(
                        REQUEST_PROJECT_NAME, quantum.QuantumProgram(), REQUEST_JOB
                    )
                manager.stop()

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_cancel_future_expects_engine_cancellation_rpc_call(self, client_constructor):
        async def test():
            async with duet.timeout_scope(5):
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=[]
                )
                manager = StreamManager(fake_client)

                result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB)
                result_future.cancel()
                await duet.sleep(1)  # Let cancellation complete asynchronously
                manager.stop()

                assert len(fake_client.cancel_requests) == 1
                assert fake_client.cancel_requests[0] == quantum.CancelQuantumJobRequest(
                    name='projects/proj/programs/prog/jobs/job0'
                )

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_stream_broken_twice_expects_retry_with_get_quantum_result_twice(
        self, client_constructor
    ):
        async def test():
            async with duet.timeout_scope(5):
                expected_result = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job0'
                )
                mock_responses_and_exceptions = [
                    google_exceptions.ServiceUnavailable('unavailable'),
                    google_exceptions.ServiceUnavailable('unavailable'),
                    quantum.QuantumRunStreamResponse(result=expected_result),
                ]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses_and_exceptions
                )
                manager = StreamManager(fake_client)

                actual_result = await manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]
                assert 'get_quantum_result' in fake_client.stream_requests[1]
                assert 'get_quantum_result' in fake_client.stream_requests[2]

        duet.run(test)

    @pytest.mark.parametrize(
        'error',
        [
            google_exceptions.InternalServerError('server error'),
            google_exceptions.ServiceUnavailable('unavailable'),
        ],
    )
    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_with_retryable_stream_breakage_expects_get_result_request(
        self, client_constructor, error
    ):
        async def test():
            async with duet.timeout_scope(5):
                mock_responses = [
                    error,
                    quantum.QuantumRunStreamResponse(
                        result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
                    ),
                ]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses
                )
                manager = StreamManager(fake_client)

                await manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB)
                manager.stop()

                assert len(fake_client.stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]
                assert 'get_quantum_result' in fake_client.stream_requests[1]

        duet.run(test)

    @pytest.mark.parametrize(
        'error',
        [
            # Including errors which are likely to occur.
            google_exceptions.DeadlineExceeded('deadline exceeded'),
            google_exceptions.FailedPrecondition('failed precondition'),
            google_exceptions.Forbidden('forbidden'),
            google_exceptions.InvalidArgument('invalid argument'),
            google_exceptions.ResourceExhausted('resource exhausted'),
            google_exceptions.TooManyRequests('too many requests'),
            google_exceptions.Unauthenticated('unauthenticated'),
            google_exceptions.Unauthorized('unauthorized'),
            google_exceptions.Unknown('unknown'),
        ],
    )
    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_with_non_retryable_stream_breakage_raises_error(
        self, client_constructor, error
    ):
        async def test():
            async with duet.timeout_scope(5):
                mock_responses = [
                    error,
                    quantum.QuantumRunStreamResponse(
                        result=quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
                    ),
                ]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses
                )
                manager = StreamManager(fake_client)

                with pytest.raises(type(error)):
                    await manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB)
                manager.stop()

                assert len(fake_client.stream_requests) == 1
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_expects_job_response(self, client_constructor):
        async def test():
            async with duet.timeout_scope(5):
                expected_job = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
                mock_responses = [quantum.QuantumRunStreamResponse(job=expected_job)]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses
                )
                manager = StreamManager(fake_client)

                actual_job = await manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                manager.stop()

                assert actual_job == expected_job
                assert len(fake_client.stream_requests) == 1
                # assert that the first request is a CreateQuantumProgramAndJobRequest.
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_job_does_not_exist_expects_create_quantum_job_request(self, client_constructor):
        async def test():
            async with duet.timeout_scope(5):
                expected_result = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job0'
                )
                mock_responses_and_exceptions = [
                    google_exceptions.ServiceUnavailable('unavailable'),
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)
                    ),
                    quantum.QuantumRunStreamResponse(result=expected_result),
                ]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses_and_exceptions
                )
                manager = StreamManager(fake_client)

                actual_result = await manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]
                assert 'get_quantum_result' in fake_client.stream_requests[1]
                assert 'create_quantum_job' in fake_client.stream_requests[2]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_does_not_exist_expects_create_quantum_program_and_job_request(
        self, client_constructor
    ):
        async def test():
            async with duet.timeout_scope(5):
                expected_result = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job0'
                )
                mock_responses_and_exceptions = [
                    google_exceptions.ServiceUnavailable('unavailable'),
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)
                    ),
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(
                            code=quantum.StreamError.Code.PROGRAM_DOES_NOT_EXIST
                        )
                    ),
                    quantum.QuantumRunStreamResponse(result=expected_result),
                ]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses_and_exceptions
                )
                manager = StreamManager(fake_client)

                actual_result = await manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.stream_requests) == 4
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]
                assert 'get_quantum_result' in fake_client.stream_requests[1]
                assert 'create_quantum_job' in fake_client.stream_requests[2]
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[3]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_already_exists_expects_program_already_exists_error(
        self, client_constructor
    ):
        async def test():
            async with duet.timeout_scope(5):
                mock_responses_and_exceptions = [
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(
                            code=quantum.StreamError.Code.PROGRAM_ALREADY_EXISTS
                        )
                    )
                ]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses_and_exceptions
                )
                manager = StreamManager(fake_client)

                with pytest.raises(ProgramAlreadyExistsError):
                    await manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB)
                manager.stop()

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_twice_in_parallel_expect_result_responses(self, client_constructor):
        async def test():
            async with duet.timeout_scope(5):
                request_job1 = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job1')
                expected_result0 = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job0'
                )
                expected_result1 = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job1'
                )
                mock_responses = [
                    quantum.QuantumRunStreamResponse(result=expected_result0),
                    quantum.QuantumRunStreamResponse(result=expected_result1),
                ]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses
                )
                manager = StreamManager(fake_client)

                actual_result0_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                actual_result1_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, request_job1
                )
                actual_result1 = await actual_result1_future
                actual_result0 = await actual_result0_future
                manager.stop()

                assert actual_result0 == expected_result0
                assert actual_result1 == expected_result1
                assert len(fake_client.stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[1]

        duet.run(test)

    # TODO(#5996) Update fake client implementation to support this test case.
    # @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    # def test_submit_twice_and_break_stream_expect_result_responses(self, client_constructor):
    #     async def test():
    #         async with duet.timeout_scope(5):
    #             request_job1 = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job1')
    #             expected_result0 = quantum.QuantumResult(
    #                 parent='projects/proj/programs/prog/jobs/job0'
    #             )
    #             expected_result1 = quantum.QuantumResult(
    #                 parent='projects/proj/programs/prog/jobs/job1'
    #             )
    #             # TODO the current fake client doesn't have the response timing flexibility
    #             # required by this test.
    #             # Ideally, the client raises ServiceUnavailable after both initial requests are
    #             # sent.
    #             mock_responses = [
    #                 google_exceptions.ServiceUnavailable('unavailable'),
    #                 google_exceptions.ServiceUnavailable('unavailable'),
    #                 quantum.QuantumRunStreamResponse(result=expected_result0),
    #                 quantum.QuantumRunStreamResponse(result=expected_result1),
    #             ]
    #             fake_client = setup_fake_quantum_run_stream_client(
    #                 client_constructor, responses_and_exceptions=mock_responses
    #             )
    #             manager = StreamManager(fake_client)

    #             actual_result0_future = manager.submit(
    #                 REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
    #             )
    #             actual_result1_future = manager.submit(
    #                 REQUEST_PROJECT_NAME, REQUEST_PROGRAM, request_job1
    #             )
    #             actual_result1 = await actual_result1_future
    #             actual_result0 = await actual_result0_future
    #             manager.stop()

    #             assert actual_result0 == expected_result0
    #             assert actual_result1 == expected_result1
    #             assert len(fake_client.stream_requests) == 2
    #             assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]
    #             assert 'create_quantum_program_and_job' in fake_client.stream_requests[1]

    #     duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_cancels_existing_sends(self, client_constructor):
        async def test():
            async with duet.timeout_scope(5):
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=[]
                )
                manager = StreamManager(fake_client)

                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                # Wait for the manager to submit a request. If request submission runs after stop(),
                # it will start the manager again and the test will block waiting for a response.
                await duet.sleep(1)
                manager.stop()

                with pytest.raises(concurrent.futures.CancelledError):
                    await actual_result_future

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_then_send_expects_result_response(self, client_constructor):
        """New requests should work after stopping the manager."""

        async def test():
            async with duet.timeout_scope(5):
                expected_result = quantum.QuantumResult(
                    parent='projects/proj/programs/prog/jobs/job0'
                )
                mock_responses = [quantum.QuantumRunStreamResponse(result=expected_result)]
                fake_client = setup_fake_quantum_run_stream_client(
                    client_constructor, responses_and_exceptions=mock_responses
                )
                manager = StreamManager(fake_client)

                manager.stop()
                actual_result = await manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB
                )
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.stream_requests) == 1
                # assert that the first request is a CreateQuantumProgramAndJobRequest.
                assert 'create_quantum_program_and_job' in fake_client.stream_requests[0]

        duet.run(test)

    @pytest.mark.parametrize(
        'error_code, current_request_type',
        [
            (Code.PROGRAM_DOES_NOT_EXIST, 'create_quantum_program_and_job'),
            (Code.PROGRAM_DOES_NOT_EXIST, 'get_quantum_result'),
            (Code.PROGRAM_ALREADY_EXISTS, 'create_quantum_job'),
            (Code.PROGRAM_ALREADY_EXISTS, 'get_quantum_result'),
            (Code.JOB_DOES_NOT_EXIST, 'create_quantum_program_and_job'),
            (Code.JOB_DOES_NOT_EXIST, 'create_quantum_job'),
        ],
    )
    def test_get_retry_request_or_raise_expects_stream_error(
        self, error_code, current_request_type
    ):
        # This tests a private function, but it's much easier to exhaustively test this function
        # than to get the stream manager to issue specific requests required for each test case.

        create_quantum_program_and_job_request = quantum.QuantumRunStreamRequest(
            create_quantum_program_and_job=quantum.CreateQuantumProgramAndJobRequest()
        )
        create_quantum_job_request = quantum.QuantumRunStreamRequest(
            create_quantum_job=quantum.CreateQuantumJobRequest()
        )
        get_quantum_result_request = quantum.QuantumRunStreamRequest(
            get_quantum_result=quantum.GetQuantumResultRequest()
        )
        if current_request_type == 'create_quantum_program_and_job':
            current_request = create_quantum_program_and_job_request
        elif current_request_type == 'create_quantum_job':
            current_request = create_quantum_job_request
        elif current_request_type == 'get_quantum_result':
            current_request = get_quantum_result_request

        with pytest.raises(StreamError):
            _get_retry_request_or_raise(
                quantum.StreamError(code=error_code),
                current_request,
                create_quantum_program_and_job_request,
                create_quantum_job_request,
            )

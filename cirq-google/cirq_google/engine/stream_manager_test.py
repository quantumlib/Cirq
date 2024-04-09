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

from typing import AsyncIterable, AsyncIterator, Awaitable, List, Sequence, Union
import asyncio
import concurrent
from unittest import mock

import duet
import pytest
import google.api_core.exceptions as google_exceptions

from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine.stream_manager import (
    _get_retry_request_or_raise,
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
REQUEST_JOB0 = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
REQUEST_JOB1 = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job1')


def setup_client(client_constructor):
    fake_client = FakeQuantumRunStream()
    client_constructor.return_value = fake_client
    return fake_client


def setup(client_constructor):
    fake_client = setup_client(client_constructor)
    return fake_client, StreamManager(fake_client)


class FakeQuantumRunStream:
    """A fake Quantum Engine client which supports QuantumRunStream and CancelQuantumJob."""

    _REQUEST_STOPPED = 'REQUEST_STOPPED'

    def __init__(self) -> None:
        self.all_stream_requests: List[quantum.QuantumRunStreamRequest] = []
        self.all_cancel_requests: List[quantum.CancelQuantumJobRequest] = []
        self._executor = AsyncioExecutor.instance()
        self._request_buffer = duet.AsyncCollector[quantum.QuantumRunStreamRequest]()
        self._request_iterator_stopped: duet.AwaitableFuture[None] = duet.AwaitableFuture()
        # asyncio.Queue needs to be initialized inside the asyncio thread because all callers need
        # to use the same event loop.
        self._responses_and_exceptions_future: duet.AwaitableFuture[
            asyncio.Queue[Union[quantum.QuantumRunStreamResponse, BaseException]]
        ] = duet.AwaitableFuture()

    async def quantum_run_stream(
        self, requests: AsyncIterator[quantum.QuantumRunStreamRequest], **kwargs
    ) -> Awaitable[AsyncIterable[quantum.QuantumRunStreamResponse]]:
        """Fakes the QuantumRunStream RPC.

        Once a request is received, it is appended to `all_stream_requests`, and the test calling
        `wait_for_requests()` is notified.

        The response is sent when a test calls `reply()` with a `QuantumRunStreamResponse`. If a
        test calls `reply()` with an exception, it is raised here to the `quantum_run_stream()`
        caller.

        This is called from the asyncio thread.
        """
        responses_and_exceptions: asyncio.Queue[
            Union[quantum.QuantumRunStreamResponse, BaseException]
        ] = asyncio.Queue()
        self._responses_and_exceptions_future.try_set_result(responses_and_exceptions)

        async def read_requests():
            async for request in requests:
                self.all_stream_requests.append(request)
                self._request_buffer.add(request)
            await responses_and_exceptions.put(FakeQuantumRunStream._REQUEST_STOPPED)
            self._request_iterator_stopped.try_set_result(None)

        async def response_iterator():
            asyncio.create_task(read_requests())
            while (
                message := await responses_and_exceptions.get()
            ) != FakeQuantumRunStream._REQUEST_STOPPED:
                if isinstance(message, quantum.QuantumRunStreamResponse):
                    yield message
                else:  # isinstance(message, BaseException)
                    self._responses_and_exceptions_future = duet.AwaitableFuture()
                    raise message

        return response_iterator()

    async def cancel_quantum_job(self, request: quantum.CancelQuantumJobRequest) -> None:
        """Records the cancellation in `cancel_requests`.

        This is called from the asyncio thread.
        """
        self.all_cancel_requests.append(request)
        await asyncio.sleep(0)

    async def wait_for_requests(self, num_requests=1) -> Sequence[quantum.QuantumRunStreamRequest]:
        """Wait til `num_requests` number of requests are received via `quantum_run_stream()`.

        This must be called from the duet thread.

        Returns:
            The received requests.
        """
        requests = []
        for _ in range(num_requests):
            requests.append(await self._request_buffer.__anext__())
        return requests

    async def reply(
        self, response_or_exception: Union[quantum.QuantumRunStreamResponse, BaseException]
    ):
        """Sends a response or raises an exception to the `quantum_run_stream()` caller.

        If input response is missing `message_id`, it is defaulted to the `message_id` of the most
        recent request. This is to support the most common use case of responding immediately after
        a request.

        Assumes that at least one request must have been submitted to the StreamManager.

        This must be called from the duet thread.
        """
        responses_and_exceptions = await self._responses_and_exceptions_future
        if (
            isinstance(response_or_exception, quantum.QuantumRunStreamResponse)
            and not response_or_exception.message_id
        ):
            response_or_exception.message_id = self.all_stream_requests[-1].message_id

        async def send():
            await responses_and_exceptions.put(response_or_exception)

        await self._executor.submit(send)

    async def wait_for_request_iterator_stop(self):
        """Wait for the request iterator to stop.

        This must be called from a duet thread.
        """
        await self._request_iterator_stopped
        self._request_iterator_stopped = duet.AwaitableFuture()


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
        # Arrange
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                # Act
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                # Assert
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 1
                # assert that the first request is a CreateQuantumProgramAndJobRequest.
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_without_name_raises(self, client_constructor):
        _, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                with pytest.raises(ValueError, match='Program name must be set'):
                    await manager.submit(
                        REQUEST_PROJECT_NAME, quantum.QuantumProgram(), REQUEST_JOB0
                    )
                manager.stop()

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_cancel_future_expects_engine_cancellation_rpc_call(self, client_constructor):
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                result_future.cancel()
                await duet.sleep(1)  # Let cancellation complete asynchronously
                manager.stop()

                assert len(fake_client.all_cancel_requests) == 1
                assert fake_client.all_cancel_requests[0] == quantum.CancelQuantumJobRequest(
                    name='projects/proj/programs/prog/jobs/job0'
                )

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_stream_broken_twice_expects_retry_with_get_quantum_result_twice(
        self, client_constructor
    ):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'get_quantum_result' in fake_client.all_stream_requests[2]

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
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(error)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                await actual_result_future
                manager.stop()

                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]

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
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(error)
                with pytest.raises(type(error)):
                    await actual_result_future
                manager.stop()

                assert len(fake_client.all_stream_requests) == 1
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_expects_job_response(self, client_constructor):
        expected_job = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_job_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(job=expected_job))
                actual_job = await actual_job_future
                manager.stop()

                assert actual_job == expected_job
                assert len(fake_client.all_stream_requests) == 1
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_job_does_not_exist_expects_create_quantum_job_request(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_does_not_exist_expects_create_quantum_program_and_job_request(
        self, client_constructor
    ):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(
                            code=quantum.StreamError.Code.PROGRAM_DOES_NOT_EXIST
                        )
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 4
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[3]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_already_exists_expects_get_result_request(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(
                            code=quantum.StreamError.Code.PROGRAM_ALREADY_EXISTS
                        )
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_already_exists_but_job_does_not_exist_expects_create_job_request(
        self, client_constructor
    ):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(
                            code=quantum.StreamError.Code.PROGRAM_ALREADY_EXISTS
                        )
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_job_already_exist_expects_get_result_request(self, client_constructor):
        """Verifies the behavior when the client receives a JOB_ALREADY_EXISTS error.

        This error is only expected to be triggered in the following race condition:
        1. The client sends a CreateQuantumProgramAndJobRequest.
        2. The client's stream disconnects.
        3. The client retries with a new stream and a GetQuantumResultRequest.
        4. The job doesn't exist yet, and the client receives a "job not found" error.
        5. Scheduler creates the program and job.
        6. The client retries with a CreateJobRequest and fails with a "job already exists" error.

        The JOB_ALREADY_EXISTS error from `CreateQuantumJobRequest` is only possible if the job
        doesn't exist yet at the last `GetQuantumResultRequest`.
        """
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                # Trigger a retry with `CreateQuantumJobRequest`.
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        error=quantum.StreamError(code=quantum.StreamError.Code.JOB_ALREADY_EXISTS)
                    )
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 4
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]
                assert 'get_quantum_result' in fake_client.all_stream_requests[3]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_twice_in_parallel_expect_result_responses(self, client_constructor):
        expected_result0 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        expected_result1 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result0_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                actual_result1_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB1
                )
                await fake_client.wait_for_requests(num_requests=2)
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=fake_client.all_stream_requests[0].message_id,
                        result=expected_result0,
                    )
                )
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=fake_client.all_stream_requests[1].message_id,
                        result=expected_result1,
                    )
                )
                actual_result1 = await actual_result1_future
                actual_result0 = await actual_result0_future
                manager.stop()

                assert actual_result0 == expected_result0
                assert actual_result1 == expected_result1
                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[1]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_twice_and_break_stream_expect_result_responses(self, client_constructor):
        expected_result0 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        expected_result1 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result0_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                actual_result1_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB1
                )
                await fake_client.wait_for_requests(num_requests=2)
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests(num_requests=2)
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=next(
                            req.message_id
                            for req in fake_client.all_stream_requests[2:]
                            if req.get_quantum_result.parent == expected_result0.parent
                        ),
                        result=expected_result0,
                    )
                )
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=next(
                            req.message_id
                            for req in fake_client.all_stream_requests[2:]
                            if req.get_quantum_result.parent == expected_result1.parent
                        ),
                        result=expected_result1,
                    )
                )
                actual_result0 = await actual_result0_future
                actual_result1 = await actual_result1_future
                manager.stop()

                assert actual_result0 == expected_result0
                assert actual_result1 == expected_result1
                assert len(fake_client.all_stream_requests) == 4
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[1]
                assert 'get_quantum_result' in fake_client.all_stream_requests[2]
                assert 'get_quantum_result' in fake_client.all_stream_requests[3]

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_cancels_existing_sends(self, client_constructor):
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                # Wait for the manager to submit a request. If request submission runs after stop(),
                # it will start the manager again and the test will block waiting for a response.
                await fake_client.wait_for_requests()
                manager.stop()

                with pytest.raises(concurrent.futures.CancelledError):
                    await actual_result_future

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_then_send_expects_result_response(self, client_constructor):
        """New requests should work after stopping the manager."""
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                manager.stop()
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()

                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 1
                # assert that the first request is a CreateQuantumProgramAndJobRequest.
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]

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
            (Code.JOB_ALREADY_EXISTS, 'get_quantum_result'),
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
                get_quantum_result_request,
            )

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_broken_stream_stops_request_iterator(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=fake_client.all_stream_requests[0].message_id,
                        result=expected_result,
                    )
                )
                await actual_result_future
                await fake_client.reply(google_exceptions.ServiceUnavailable('service unavailable'))
                await fake_client.wait_for_request_iterator_stop()
                manager.stop()

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_stops_request_iterator(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=fake_client.all_stream_requests[0].message_id,
                        result=expected_result,
                    )
                )
                await actual_result_future
                manager.stop()
                await fake_client.wait_for_request_iterator_stop()

        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_after_stream_breakage(self, client_constructor):
        expected_result0 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        expected_result1 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result0_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=fake_client.all_stream_requests[0].message_id,
                        result=expected_result0,
                    )
                )
                actual_result0 = await actual_result0_future
                await fake_client.reply(google_exceptions.ServiceUnavailable('service unavailable'))
                actual_result1_future = manager.submit(
                    REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0
                )
                await fake_client.wait_for_requests()
                await fake_client.reply(
                    quantum.QuantumRunStreamResponse(
                        message_id=fake_client.all_stream_requests[1].message_id,
                        result=expected_result1,
                    )
                )
                actual_result1 = await actual_result1_future
                manager.stop()

                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[1]
                assert actual_result0 == expected_result0
                assert actual_result1 == expected_result1

        duet.run(test)

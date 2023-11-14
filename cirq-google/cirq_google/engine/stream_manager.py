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

from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet

import google.api_core.exceptions as google_exceptions

from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor

Code = quantum.StreamError.Code


RETRYABLE_GOOGLE_API_EXCEPTIONS = [
    google_exceptions.InternalServerError,
    google_exceptions.ServiceUnavailable,
]


class ProgramAlreadyExistsError(Exception):
    def __init__(self, program_name: str):
        # Call the base class constructor with the parameters it needs
        super().__init__(f"'{program_name}' already exists")


class StreamError(Exception):
    pass


class ResponseDemux:
    """An event demultiplexer for QuantumRunStreamResponses, as part of the async reactor pattern.

    A caller can subscribe to the response matching a provided message ID. Only a single caller may
    subscribe to each ID.

    Another caller can subsequently publish a response. The future held by the subscriber with
    the matching message ID will then be fulfilled.

    A caller can also publish an exception to all subscribers.
    """

    def __init__(self) -> None:
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

    def publish_exception(self, exception: BaseException) -> None:
        """Publishes an exception to all outstanding futures."""
        for future in self._subscribers.values():
            if not future.done():
                future.set_exception(exception)
        self._subscribers.clear()


class StreamManager:
    """Manages communication with Quantum Engine via QuantumRunStream, a bi-directional stream RPC.

    The main manager method is `submit()`, which sends the provided job to Quantum Engine through
    the stream and returns a future to be completed when either the result is ready or the job has
    failed. The submitted job can also be cancelled by calling `cancel()` on the future returned by
    `submit()`.

    A new stream is opened during the first `submit()` call, and it stays open. If the stream is
    unused, users can close the stream and free management resources by calling `stop()`.

    """

    def __init__(self, grpc_client: quantum.QuantumEngineServiceAsyncClient):
        self._grpc_client = grpc_client
        # Used to determine whether the stream coroutine is actively running, and provides a way to
        # cancel it.
        self._manage_stream_loop_future: Optional[duet.AwaitableFuture[None]] = None
        # TODO(#5996) consider making the scope of response futures local to the relevant tasks
        # rather than all of StreamManager.
        # Currently, this field is being written to from both duet and asyncio threads. While the
        # ResponseDemux implementation does support this, it does not guarantee thread safety in its
        # interface.
        self._response_demux = ResponseDemux()
        self._next_available_message_id = 0
        # Construct queue in AsyncioExecutor to ensure it binds to the correct event loop, since it
        # is used by asyncio coroutines.
        self._request_queue = self._executor.submit(self._make_request_queue).result()

    async def _make_request_queue(self) -> asyncio.Queue[Optional[quantum.QuantumRunStreamRequest]]:
        """Returns a queue used to back the request iterator passed to the stream.

        If `None` is put into the queue, the request iterator will stop.
        """
        return asyncio.Queue()

    def submit(
        self, project_name: str, program: quantum.QuantumProgram, job: quantum.QuantumJob
    ) -> duet.AwaitableFuture[Union[quantum.QuantumResult, quantum.QuantumJob]]:
        """Submits a job over the stream and returns a future for the result.

        If submit() is called for the first time since StreamManager instantiation or since the last
        time stop() was called, it will create a new long-running stream.

        The job can be cancelled by calling `cancel()` on the returned future.

        Args:
            project_name: The full project ID resource path associated with the job.
            program: The Quantum Engine program representing the circuit to be executed. The program
                name must be set.
            job: The Quantum Engine job to be executed.

        Returns:
            A future for the job result, or the job if the job has failed.

        Raises:
            ProgramAlreadyExistsError: if the program already exists.
            StreamError: if there is a non-retryable error while executing the job.
            ValueError: if program name is not set.
            concurrent.futures.CancelledError: if the stream is stopped while a job is in flight.
            google.api_core.exceptions.GoogleAPICallError: if the stream breaks with a non-retryable
                error.
        """
        if 'name' not in program:
            raise ValueError('Program name must be set.')

        if self._manage_stream_loop_future is None or self._manage_stream_loop_future.done():
            self._manage_stream_loop_future = self._executor.submit(
                self._manage_stream, self._request_queue
            )
        return self._executor.submit(
            self._manage_execution, self._request_queue, project_name, program, job
        )

    def stop(self) -> None:
        """Closes the open stream and resets all management resources."""
        if (
            self._manage_stream_loop_future is not None
            and not self._manage_stream_loop_future.done()
        ):
            self._manage_stream_loop_future.cancel()
        self._response_demux.publish_exception(asyncio.CancelledError())
        self._reset()

    def _reset(self):
        """Resets the manager state."""
        self._manage_stream_loop_future = None
        self._response_demux = ResponseDemux()
        self._request_queue = self._executor.submit(self._make_request_queue).result()

    @property
    def _executor(self) -> AsyncioExecutor:
        # We must re-use a single Executor due to multi-threading issues in gRPC
        # clients: https://github.com/grpc/grpc/issues/25364.
        return AsyncioExecutor.instance()

    async def _manage_stream(
        self, request_queue: asyncio.Queue[Optional[quantum.QuantumRunStreamRequest]]
    ) -> None:
        """The stream coroutine, an asyncio coroutine to manage QuantumRunStream.

        This coroutine reads responses from the stream and forwards them to the ResponseDemux, where
        the corresponding execution coroutine `_manage_request()` is notified.

        When the stream breaks, the stream is reopened, and all execution coroutines are notified.

        There is at most a single instance of this coroutine running.

        Args:
            request_queue: The queue holding requests from the execution coroutine.
        """
        while True:
            try:
                # The default gRPC client timeout is used.
                response_iterable = await self._grpc_client.quantum_run_stream(
                    _request_iterator(request_queue)
                )
                async for response in response_iterable:
                    self._response_demux.publish(response)
            except asyncio.CancelledError:
                await request_queue.put(None)
                break
            except BaseException as e:
                # Note: the message ID counter is not reset upon a new stream.
                await request_queue.put(None)
                self._response_demux.publish_exception(e)  # Raise to all request tasks

    async def _manage_execution(
        self,
        request_queue: asyncio.Queue[Optional[quantum.QuantumRunStreamRequest]],
        project_name: str,
        program: quantum.QuantumProgram,
        job: quantum.QuantumJob,
    ) -> Union[quantum.QuantumResult, quantum.QuantumJob]:
        """The execution coroutine, an asyncio coroutine to manage the lifecycle of a job execution.

        This coroutine sends QuantumRunStream requests to the request iterator and receives
        responses from the ResponseDemux.

        It initially sends a CreateQuantumProgramAndJobRequest, and retries if there is a retryable
        error by sending another request. The exact request type depends on the error.

        There is one execution coroutine per running job submission.

        Args:
            request_queue: The queue used to send requests to the stream coroutine.
            project_name: The full project ID resource path associated with the job.
            program: The Quantum Engine program representing the circuit to be executed.
            job: The Quantum Engine job to be executed.

        Raises:
            concurrent.futures.CancelledError: if either the request is cancelled or the stream
                coroutine is cancelled.
            google.api_core.exceptions.GoogleAPICallError: if the stream breaks with a non-retryable
                error.
            ValueError: if the response is of a type which is not recognized by this client.
        """
        create_program_and_job_request = quantum.QuantumRunStreamRequest(
            parent=project_name,
            create_quantum_program_and_job=quantum.CreateQuantumProgramAndJobRequest(
                parent=project_name, quantum_program=program, quantum_job=job
            ),
        )

        current_request = create_program_and_job_request
        while True:
            try:
                current_request.message_id = self._generate_message_id()
                response_future = self._response_demux.subscribe(current_request.message_id)
                await request_queue.put(current_request)
                response = await response_future

            # Broken stream
            except google_exceptions.GoogleAPICallError as e:
                if not _is_retryable_error(e):
                    raise e

                # Retry
                current_request = _to_get_result_request(create_program_and_job_request)
                continue
                # TODO(#5996) add exponential backoff

            # Either when this request is canceled or the _manage_stream() loop is canceled.
            except asyncio.CancelledError:
                # TODO(#5996) Consider moving the request future cancellation logic into a future
                # done callback, so that the the cancellation caller can wait for it to complete.
                # TODO(#5996) Check the condition that response_future is not done before
                # cancelling, once request cancellation is moved to a callback.
                if response_future is not None:
                    response_future.cancel()
                    await self._cancel(job.name)
                raise

            # Response handling
            if 'result' in response:
                return response.result
            elif 'job' in response:
                return response.job
            elif 'error' in response:
                current_request = _get_retry_request_or_raise(
                    response.error,
                    current_request,
                    create_program_and_job_request,
                    _to_create_job_request(create_program_and_job_request),
                )
                continue
            else:  # pragma: no cover
                raise ValueError(
                    'The Quantum Engine response type is not recognized by this client. '
                    'This may be due to an outdated version of cirq-google'
                )

    async def _cancel(self, job_name: str) -> None:
        await self._grpc_client.cancel_quantum_job(quantum.CancelQuantumJobRequest(name=job_name))

    def _generate_message_id(self) -> str:
        message_id = str(self._next_available_message_id)
        self._next_available_message_id += 1
        return message_id


def _get_retry_request_or_raise(
    error: quantum.StreamError,
    current_request,
    create_program_and_job_request,
    create_job_request: quantum.QuantumRunStreamRequest,
):
    """Decide whether the given stream error is retryable.

    If it is, returns the next stream request to send upon retry. Otherwise, raises an error.
    """
    if error.code == Code.PROGRAM_DOES_NOT_EXIST:
        if 'create_quantum_job' in current_request:
            return create_program_and_job_request
    elif error.code == Code.PROGRAM_ALREADY_EXISTS:
        if 'create_quantum_program_and_job' in current_request:
            raise ProgramAlreadyExistsError(
                current_request.create_quantum_program_and_job.quantum_program.name
            )
    elif error.code == Code.JOB_DOES_NOT_EXIST:
        if 'get_quantum_result' in current_request:
            return create_job_request

    # Code.JOB_ALREADY_EXISTS should never happen.
    # The first stream request is always a CreateQuantumProgramAndJobRequest, which never fails
    # with this error because jobs are scoped within a program.
    # CreateQuantumJobRequests would fail with a PROGRAM_ALREADY_EXISTS if the job already
    # exists because program and job creation happen atomically for a
    # CreateQuantumProgramAndJobRequest.

    raise StreamError(error.message)


def _is_retryable_error(e: google_exceptions.GoogleAPICallError) -> bool:
    return any(isinstance(e, exception_type) for exception_type in RETRYABLE_GOOGLE_API_EXCEPTIONS)


async def _request_iterator(
    request_queue: asyncio.Queue[Optional[quantum.QuantumRunStreamRequest]],
) -> AsyncIterator[quantum.QuantumRunStreamRequest]:
    """The request iterator for Quantum Engine client RPC quantum_run_stream().

    Every call to this method generates a new iterator.
    """
    while request := await request_queue.get():
        yield request


def _to_create_job_request(
    create_program_and_job_request: quantum.QuantumRunStreamRequest,
) -> quantum.QuantumRunStreamRequest:
    """Converted the QuantumRunStreamRequest from a CreateQuantumProgramAndJobRequest to a
    CreateQuantumJobRequest.
    """
    program = create_program_and_job_request.create_quantum_program_and_job.quantum_program
    job = create_program_and_job_request.create_quantum_program_and_job.quantum_job
    return quantum.QuantumRunStreamRequest(
        parent=create_program_and_job_request.parent,
        create_quantum_job=quantum.CreateQuantumJobRequest(parent=program.name, quantum_job=job),
    )


def _to_get_result_request(
    create_program_and_job_request: quantum.QuantumRunStreamRequest,
) -> quantum.QuantumRunStreamRequest:
    """Converted the QuantumRunStreamRequest from a CreateQuantumProgramAndJobRequest to a
    GetQuantumResultRequest.
    """
    job = create_program_and_job_request.create_quantum_program_and_job.quantum_job
    return quantum.QuantumRunStreamRequest(
        parent=create_program_and_job_request.parent,
        get_quantum_result=quantum.GetQuantumResultRequest(parent=job.name),
    )

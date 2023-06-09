from collections.abc import Callable

import asyncio
import duet

from cirq_google.cloud import quantum

JobPath = str
MessageId = str
CancelCallback = Callable[[quantum.QuantumRunStreamRequest], None]


class _StreamResponseDemux:
    """A event demultiplexer for QuantumRunStreamResponses, as part of the async reactor pattern.

    Args:
        cancel_callback: Function to be called when the future matching its request argument is
        canceled.
    TODO(#5996) expand
    """

    def __init__(self, cancel_callback: CancelCallback):
        self._subscribers: dict[JobPath, dict[MessageId, duet.AwaitableFuture]] = {}
        self._cancel_callback = cancel_callback

    def subscribe(
        self, request: quantum.QuantumRunStreamRequest
    ) -> duet.AwaitableFuture[quantum.QuantumRunStreamResponse]:
        """Subscribes to the QuantumRunStreamResponse matching the given request.

        The request must be unique among all calls to this method, with message ID as the
        identifier.

        Returns:
            A future for the response to be fulfilled when publish is called.
        Raises:
            ValueError: If there is already a subscriber for the given request.
        """

        request_type = request.WhichOneOf('request')
        if request_type == 'create_quantum_program_and_job':
            job_path = request.create_quantum_program_and_job.quantum_job.name
        elif request_type == 'create_quantum_job':
            job_path = request.create_quantum_job.quantum_job.name
        else:  # request_type == 'get_quantum_result'
            job_path = request.get_quantum_result.parent

        if job_path not in self._subscribers:
            self._subscribers[job_path] = {}

        subscribers_by_message = self._subscribers[job_path]
        if request.message_id in subscribers_by_message:
            raise ValueError(f'There is another subscriber for the message ID {request.message_id}')

        def cancel(future: asyncio.Future):
            if future.cancelled():
                self._cancel_callback(request)

        response_future = duet.AwaitableFuture[quantum.QuantumRunStreamResponse]()
        response_future.add_done_callback(cancel)
        self._subscribers[job_path][request.message_id] = response_future
        return response_future

    def publish(self, response: quantum.QuantumRunStreamResponse) -> None:
        """Makes the response available to all appropriate subscribers and unsubscribes them.

        If the response has type QuantumResult or QuantumJob, all subscribers to the matching job
        path are notified and unsubscribed. If there are no subscribers matching the job path, this
        method does nothing.
        If the response has type StreamError, only the subscriber with the matching message ID is
        notified and unsubscribed. If there are no subscribers matching the message ID, this method
        does nothing.
        """

        response_type = response.WhichOneOf('response')
        if response_type == 'error':
            for subscribers_by_message in self._subscribers.values():
                if response.message_id in subscribers_by_message:
                    subscribers_by_message[response.message_id].try_set_result(response)
                    del subscribers_by_message[response.message_id]
                    break
            return

        if response_type == 'job':
            job_path = response.job.name
        else:  # response_type == 'result':
            job_path = response.result.parent

        if job_path not in self._subscribers:
            return

        for response_future in self._subscribers[job_path].values():
            response_future.try_set_result(response)
        del self._subscribers[job_path]

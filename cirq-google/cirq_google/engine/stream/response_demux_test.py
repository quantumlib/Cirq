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

from enum import Enum

import pytest

from cirq_google.engine.stream.response_demux import ResponseDemux
from cirq_google.cloud import quantum


class _RequestType(Enum):
    CREATE_QUANTUM_PROGRAM_AND_JOB = 1
    CREATE_QUANTUM_JOB = 2
    GET_QUANTUM_RESULT = 3


class _ResponseType(Enum):
    QUANTUM_RESULT = 1
    QUANTUM_JOB = 2
    STREAM_ERROR = 3


def _noop():
    pass


def _create_request(request_type, message_id, job_id):
    if request_type == _RequestType.CREATE_QUANTUM_PROGRAM_AND_JOB:
        return quantum.QuantumRunStreamRequest(
            message_id=message_id,
            parent='projects/proj',
            create_quantum_program_and_job=quantum.CreateQuantumProgramAndJobRequest(
                parent='projects/proj',
                quantum_program=quantum.QuantumProgram(name='projects/proj/programs/prog'),
                quantum_job=quantum.QuantumJob(name=f'projects/proj/programs/prog/jobs/{job_id}'),
            ),
        )
    elif request_type == _RequestType.CREATE_QUANTUM_JOB:
        return quantum.QuantumRunStreamRequest(
            message_id=message_id,
            parent='projects/proj',
            create_quantum_job=quantum.CreateQuantumJobRequest(
                parent='projects/proj/programs/prog',
                quantum_job=quantum.QuantumJob(name=f'projects/proj/programs/prog/jobs/{job_id}'),
            ),
        )
    else:
        return quantum.QuantumRunStreamRequest(
            message_id=message_id,
            parent='projects/proj',
            get_quantum_result=quantum.GetQuantumResultRequest(
                parent=f'projects/proj/programs/prog/jobs/{job_id}'
            ),
        )


def _create_response(response_type, message_id, job_id):
    if response_type == _ResponseType.QUANTUM_RESULT:
        return quantum.QuantumRunStreamResponse(
            message_id=message_id,
            result=quantum.QuantumResult(parent=f'projects/proj/programs/prog/jobs/{job_id}'),
        )
    elif response_type == _ResponseType.QUANTUM_JOB:
        return quantum.QuantumRunStreamResponse(
            message_id=message_id,
            job=quantum.QuantumJob(name=f'projects/proj/programs/prog/jobs/{job_id}'),
        )
    else:
        return quantum.QuantumRunStreamResponse(
            message_id=message_id,
            error=quantum.StreamError(code=quantum.StreamError.Code.CODE_UNSPECIFIED),
        )


def test_one_subscribe_one_publish():
    demux = ResponseDemux(cancel_callback=_noop)
    request = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')
    expected_response = _create_response(
        _ResponseType.QUANTUM_RESULT, message_id='0', job_id='job0'
    )

    future = demux.subscribe(request)
    demux.publish(expected_response)
    actual_response = future.result(timeout=1)

    assert actual_response == expected_response


def test_response_publishes_to_all_matching_job_subscribers():
    demux = ResponseDemux(cancel_callback=_noop)
    request0 = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')
    request1 = _create_request(_RequestType.GET_QUANTUM_RESULT, message_id='1', job_id='job0')
    expected_response = _create_response(
        _ResponseType.QUANTUM_RESULT, message_id='0', job_id='job0'
    )

    future0 = demux.subscribe(request0)
    future1 = demux.subscribe(request1)
    demux.publish(expected_response)
    actual_response0 = future0.result(timeout=1)
    actual_response1 = future1.result(timeout=1)

    assert actual_response0 == expected_response
    assert actual_response1 == expected_response


def test_out_of_order_response_publishes_to_all_matching_job_subscribers():
    demux = ResponseDemux(cancel_callback=_noop)
    request0 = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')
    request1 = _create_request(_RequestType.GET_QUANTUM_RESULT, message_id='1', job_id='job0')
    expected_response = _create_response(
        _ResponseType.QUANTUM_RESULT, message_id='1', job_id='job0'
    )

    future0 = demux.subscribe(request0)
    future1 = demux.subscribe(request1)
    demux.publish(expected_response)
    actual_response0 = future0.result(timeout=1)
    actual_response1 = future1.result(timeout=1)

    assert actual_response0 == expected_response
    assert actual_response1 == expected_response


def test_error_publishes_to_subscriber_with_matching_message_id():
    demux = ResponseDemux(cancel_callback=_noop)
    request0 = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')
    request1 = _create_request(_RequestType.GET_QUANTUM_RESULT, message_id='1', job_id='job0')
    expected_response = _create_response(_ResponseType.STREAM_ERROR, message_id='0', job_id='')

    future0 = demux.subscribe(request0)
    future1 = demux.subscribe(request1)
    demux.publish(expected_response)
    actual_response0 = future0.result(timeout=1)

    assert actual_response0 == expected_response
    assert not future1.done()


def test_cancel():
    cancel_called = [False]  # Using a list so that the boolean can be passed by reference.

    def cancel(_):
        cancel_called[0] = True

    demux = ResponseDemux(cancel_callback=cancel)
    request0 = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')

    future0 = demux.subscribe(request0)
    success = future0.cancel()

    assert success
    assert cancel_called[0]


def test_response_job_name_does_not_exist():
    demux = ResponseDemux(cancel_callback=_noop)
    request = _create_request(
        _RequestType.CREATE_QUANTUM_PROGRAM_AND_JOB, message_id='0', job_id='job0'
    )
    expected_response = _create_response(
        _ResponseType.QUANTUM_RESULT, message_id='0', job_id='job1'
    )

    future = demux.subscribe(request)
    demux.publish(expected_response)

    assert not future.done()


def test_error_message_id_does_not_exist():
    demux = ResponseDemux(cancel_callback=_noop)
    request = _create_request(
        _RequestType.CREATE_QUANTUM_PROGRAM_AND_JOB, message_id='0', job_id='job0'
    )
    expected_response = _create_response(_ResponseType.STREAM_ERROR, message_id='1', job_id='job0')

    future = demux.subscribe(request)
    demux.publish(expected_response)

    assert not future.done()


def test_no_subscribers():
    """Ensures publish does not throw an error when there are no subscribers at all."""
    demux = ResponseDemux(cancel_callback=_noop)
    expected_response = _create_response(
        _ResponseType.QUANTUM_RESULT, message_id='1', job_id='job0'
    )

    demux.publish(expected_response)


def test_duplicate_subscribers():
    with pytest.raises(ValueError):
        demux = ResponseDemux(cancel_callback=_noop)
        request0 = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')
        request1 = _create_request(_RequestType.GET_QUANTUM_RESULT, message_id='0', job_id='job0')
        _ = demux.subscribe(request0)

        demux.subscribe(request1)


def test_responds_twice_future_unchanged():
    demux = ResponseDemux(cancel_callback=_noop)
    request0 = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')
    request1 = _create_request(_RequestType.GET_QUANTUM_RESULT, message_id='1', job_id='job0')
    expected_response0 = _create_response(_ResponseType.QUANTUM_JOB, message_id='0', job_id='job0')
    expected_response1 = _create_response(
        _ResponseType.QUANTUM_RESULT, message_id='1', job_id='job0'
    )

    future0 = demux.subscribe(request0)
    future1 = demux.subscribe(request1)
    demux.publish(expected_response0)
    demux.publish(expected_response1)
    actual_response0 = future0.result(timeout=1)
    actual_response1 = future1.result(timeout=1)

    assert actual_response0 == expected_response0
    assert actual_response1 == expected_response0


def test_response_after_error_does_not_publish_to_errored_request():
    demux = ResponseDemux(cancel_callback=_noop)
    request0 = _create_request(_RequestType.CREATE_QUANTUM_JOB, message_id='0', job_id='job0')
    request1 = _create_request(_RequestType.GET_QUANTUM_RESULT, message_id='1', job_id='job0')
    expected_response0 = _create_response(
        _ResponseType.QUANTUM_RESULT, message_id='0', job_id='job0'
    )
    expected_response1 = _create_response(_ResponseType.STREAM_ERROR, message_id='1', job_id='')

    future0 = demux.subscribe(request0)
    future1 = demux.subscribe(request1)
    demux.publish(expected_response1)
    demux.publish(expected_response0)
    actual_response0 = future0.result(timeout=1)
    actual_response1 = future1.result(timeout=1)

    assert actual_response0 == expected_response0
    assert actual_response1 == expected_response1

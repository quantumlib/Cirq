# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Sequence

from google.api_core import gapic_v1, retry as retries, retry_async as retries_async

try:
    OptionalRetry = retries.Retry | gapic_v1.method._MethodDefault | None
    OptionalAsyncRetry = retries_async.AsyncRetry | gapic_v1.method._MethodDefault | None
except AttributeError:  # pragma: NO COVER
    OptionalRetry = retries.Retry | object | None  # type: ignore
    OptionalAsyncRetry = retries_async.AsyncRetry | object | None  # type: ignore

from cirq_google.cloud.quantum_v1alpha1.types import engine, quantum


class ListQuantumProgramsPager:
    """A pager for iterating through ``list_quantum_programs`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``programs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumPrograms`` requests and continue to iterate
    through the ``programs`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumProgramsResponse],
        request: engine.ListQuantumProgramsRequest,
        response: engine.ListQuantumProgramsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumProgramsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumProgramsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumProgram]:
        for page in self.pages:
            yield from page.programs

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumProgramsAsyncPager:
    """A pager for iterating through ``list_quantum_programs`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``programs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumPrograms`` requests and continue to iterate
    through the ``programs`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumProgramsResponse]],
        request: engine.ListQuantumProgramsRequest,
        response: engine.ListQuantumProgramsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumProgramsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumProgramsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumProgram]:
        async def async_generator():
            async for page in self.pages:
                for response in page.programs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumJobsPager:
    """A pager for iterating through ``list_quantum_jobs`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumJobs`` requests and continue to iterate
    through the ``jobs`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumJobsResponse],
        request: engine.ListQuantumJobsRequest,
        response: engine.ListQuantumJobsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumJobsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumJob]:
        for page in self.pages:
            yield from page.jobs

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumJobsAsyncPager:
    """A pager for iterating through ``list_quantum_jobs`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumJobs`` requests and continue to iterate
    through the ``jobs`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumJobsResponse]],
        request: engine.ListQuantumJobsRequest,
        response: engine.ListQuantumJobsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumJobsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumJobEventsPager:
    """A pager for iterating through ``list_quantum_job_events`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``events`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumJobEvents`` requests and continue to iterate
    through the ``events`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumJobEventsResponse],
        request: engine.ListQuantumJobEventsRequest,
        response: engine.ListQuantumJobEventsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumJobEventsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumJobEventsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumJobEvent]:
        for page in self.pages:
            yield from page.events

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumJobEventsAsyncPager:
    """A pager for iterating through ``list_quantum_job_events`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``events`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumJobEvents`` requests and continue to iterate
    through the ``events`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumJobEventsResponse]],
        request: engine.ListQuantumJobEventsRequest,
        response: engine.ListQuantumJobEventsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumJobEventsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumJobEventsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumJobEvent]:
        async def async_generator():
            async for page in self.pages:
                for response in page.events:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumProcessorsPager:
    """A pager for iterating through ``list_quantum_processors`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``processors`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumProcessors`` requests and continue to iterate
    through the ``processors`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumProcessorsResponse],
        request: engine.ListQuantumProcessorsRequest,
        response: engine.ListQuantumProcessorsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumProcessorsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumProcessorsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumProcessor]:
        for page in self.pages:
            yield from page.processors

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumProcessorsAsyncPager:
    """A pager for iterating through ``list_quantum_processors`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``processors`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumProcessors`` requests and continue to iterate
    through the ``processors`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumProcessorsResponse]],
        request: engine.ListQuantumProcessorsRequest,
        response: engine.ListQuantumProcessorsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumProcessorsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumProcessorsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumProcessor]:
        async def async_generator():
            async for page in self.pages:
                for response in page.processors:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumCalibrationsPager:
    """A pager for iterating through ``list_quantum_calibrations`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``calibrations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumCalibrations`` requests and continue to iterate
    through the ``calibrations`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumCalibrationsResponse],
        request: engine.ListQuantumCalibrationsRequest,
        response: engine.ListQuantumCalibrationsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumCalibrationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumCalibrationsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumCalibration]:
        for page in self.pages:
            yield from page.calibrations

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumCalibrationsAsyncPager:
    """A pager for iterating through ``list_quantum_calibrations`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``calibrations`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumCalibrations`` requests and continue to iterate
    through the ``calibrations`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumCalibrationsResponse]],
        request: engine.ListQuantumCalibrationsRequest,
        response: engine.ListQuantumCalibrationsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumCalibrationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumCalibrationsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumCalibration]:
        async def async_generator():
            async for page in self.pages:
                for response in page.calibrations:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumReservationsPager:
    """A pager for iterating through ``list_quantum_reservations`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``reservations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumReservations`` requests and continue to iterate
    through the ``reservations`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumReservationsResponse],
        request: engine.ListQuantumReservationsRequest,
        response: engine.ListQuantumReservationsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumReservationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumReservationsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumReservation]:
        for page in self.pages:
            yield from page.reservations

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumReservationsAsyncPager:
    """A pager for iterating through ``list_quantum_reservations`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``reservations`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumReservations`` requests and continue to iterate
    through the ``reservations`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumReservationsResponse]],
        request: engine.ListQuantumReservationsRequest,
        response: engine.ListQuantumReservationsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumReservationsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumReservationsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumReservation]:
        async def async_generator():
            async for page in self.pages:
                for response in page.reservations:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumReservationGrantsPager:
    """A pager for iterating through ``list_quantum_reservation_grants`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse` object,
    and provides an ``__iter__`` method to iterate through its ``reservation_grants`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumReservationGrants`` requests and continue to iterate
    through the ``reservation_grants`` field on the
    corresponding responses.

    All the usual
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse`
    attributes are available on the pager. If multiple requests are made, only the most recent
    response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumReservationGrantsResponse],
        request: engine.ListQuantumReservationGrantsRequest,
        response: engine.ListQuantumReservationGrantsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        self._method = method
        self._request = engine.ListQuantumReservationGrantsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumReservationGrantsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumReservationGrant]:
        for page in self.pages:
            yield from page.reservation_grants

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumReservationGrantsAsyncPager:
    """A pager for iterating through ``list_quantum_reservation_grants`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse` object,
    and provides an ``__aiter__`` method to iterate through its ``reservation_grants`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumReservationGrants`` requests and continue to iterate
    through the ``reservation_grants`` field on the
    corresponding responses.

    All the usual
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse`
    attributes are available on the pager. If multiple requests are made, only the most recent
    response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumReservationGrantsResponse]],
        request: engine.ListQuantumReservationGrantsRequest,
        response: engine.ListQuantumReservationGrantsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        self._method = method
        self._request = engine.ListQuantumReservationGrantsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumReservationGrantsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumReservationGrant]:
        async def async_generator():
            async for page in self.pages:
                for response in page.reservation_grants:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumReservationBudgetsPager:
    """A pager for iterating through ``list_quantum_reservation_budgets`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse` object,
    and provides an ``__iter__`` method to iterate through its ``reservation_budgets`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumReservationBudgets`` requests and continue to iterate
    through the ``reservation_budgets`` field on the
    corresponding responses.

    All the usual
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse`
    attributes are available on the pager. If multiple requests are made, only the most recent
    response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumReservationBudgetsResponse],
        request: engine.ListQuantumReservationBudgetsRequest,
        response: engine.ListQuantumReservationBudgetsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        self._method = method
        self._request = engine.ListQuantumReservationBudgetsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumReservationBudgetsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumReservationBudget]:
        for page in self.pages:
            yield from page.reservation_budgets

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumReservationBudgetsAsyncPager:
    """A pager for iterating through ``list_quantum_reservation_budgets`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse` object,
    and provides an ``__aiter__`` method to iterate through its ``reservation_budgets`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumReservationBudgets`` requests and continue to iterate
    through the ``reservation_budgets`` field on the
    corresponding responses.

    All the usual
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse`
    attributes are available on the pager. If multiple requests are made, only the most recent
    response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumReservationBudgetsResponse]],
        request: engine.ListQuantumReservationBudgetsRequest,
        response: engine.ListQuantumReservationBudgetsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        self._method = method
        self._request = engine.ListQuantumReservationBudgetsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumReservationBudgetsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumReservationBudget]:
        async def async_generator():
            async for page in self.pages:
                for response in page.reservation_budgets:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumTimeSlotsPager:
    """A pager for iterating through ``list_quantum_time_slots`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``time_slots`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumTimeSlots`` requests and continue to iterate
    through the ``time_slots`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., engine.ListQuantumTimeSlotsResponse],
        request: engine.ListQuantumTimeSlotsRequest,
        response: engine.ListQuantumTimeSlotsResponse,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse):
                The initial response object.
            retry (google.api_core.retry.Retry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumTimeSlotsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumTimeSlotsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumTimeSlot]:
        for page in self.pages:
            yield from page.time_slots

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)


class ListQuantumTimeSlotsAsyncPager:
    """A pager for iterating through ``list_quantum_time_slots`` requests.

    This class thinly wraps an initial
    :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``time_slots`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumTimeSlots`` requests and continue to iterate
    through the ``time_slots`` field on the
    corresponding responses.

    All the usual :class:`cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[engine.ListQuantumTimeSlotsResponse]],
        request: engine.ListQuantumTimeSlotsRequest,
        response: engine.ListQuantumTimeSlotsResponse,
        *,
        retry: OptionalAsyncRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsRequest):
                The initial request object.
            response (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse):
                The initial response object.
            retry (google.api_core.retry.AsyncRetry): Designation of what errors,
                if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        self._method = method
        self._request = engine.ListQuantumTimeSlotsRequest(request)
        self._response = response
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumTimeSlotsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(
                self._request, retry=self._retry, timeout=self._timeout, metadata=self._metadata
            )
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumTimeSlot]:
        async def async_generator():
            async for page in self.pages:
                for response in page.time_slots:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)

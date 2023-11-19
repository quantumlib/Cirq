# Copyright 2020 The Cirq Developers
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

import datetime
import sys
from typing import (
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Tuple,
    Union,
)
import warnings

import duet
import proto
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.protobuf import any_pb2, field_mask_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from cirq._compat import cached_property
from cirq._compat import deprecated_parameter
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine import stream_manager

_M = TypeVar('_M', bound=proto.Message)
_R = TypeVar('_R')


class EngineException(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


RETRYABLE_ERROR_CODES = [500, 503]


class EngineClient:
    """Client for the Quantum Engine API handling protos and gRPC client.

    This is the client for the Quantum Engine API that deals with the engine protos
    and the gRPC client but not cirq protos or objects. All users are likely better
    served by using the `Engine`, `EngineProgram`, `EngineJob`, `EngineProcessor`, and
    `Calibration` objects instead of using this directly.
    """

    def __init__(
        self,
        service_args: Optional[Dict] = None,
        verbose: Optional[bool] = None,
        max_retry_delay_seconds: int = 3600,  # 1 hour
    ) -> None:
        """Constructs a client for the Quantum Engine API.

        Args:
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying gRPC client.
            verbose: Suppresses stderr messages when set to False. Default is
                true.
            max_retry_delay_seconds: The maximum number of seconds to retry when
                a retryable error code is returned.
        """
        self.max_retry_delay_seconds = max_retry_delay_seconds
        if verbose is None:
            verbose = True
        self.verbose = verbose

        if not service_args:
            service_args = {}

        self._service_args = service_args

    @property
    def _executor(self) -> AsyncioExecutor:
        # We must re-use a single Executor due to multi-threading issues in gRPC
        # clients: https://github.com/grpc/grpc/issues/25364.
        return AsyncioExecutor.instance()

    @cached_property
    def grpc_client(self) -> quantum.QuantumEngineServiceAsyncClient:
        """Creates an async grpc client for the quantum engine service."""

        async def make_client():
            # Suppress warnings about using Application Default Credentials.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return quantum.QuantumEngineServiceAsyncClient(**self._service_args)

        return self._executor.submit(make_client).result()

    @cached_property
    def _stream_manager(self) -> stream_manager.StreamManager:
        return stream_manager.StreamManager(self.grpc_client)

    async def _send_request_async(self, func: Callable[[_M], Awaitable[_R]], request: _M) -> _R:
        """Sends a request by invoking an asyncio callable."""
        return await self._run_retry_async(func, request)

    async def _send_list_request_async(
        self, func: Callable[[_M], Awaitable[AsyncIterable[_R]]], request: _M
    ) -> List[_R]:
        """Sends a request by invoking an asyncio callable and collecting results.

        This is used for requests that return paged results. Inside the asyncio
        event loop, we iterate over all results and collect then into a list.
        """

        async def new_func(request: _M) -> List[_R]:
            pager = await func(request)
            return [item async for item in pager]

        return await self._run_retry_async(new_func, request)

    async def _run_retry_async(self, func: Callable[[_M], Awaitable[_R]], request: _M) -> _R:
        """Runs an asyncio callable and retries with exponential backoff."""
        # Start with a 100ms retry delay with exponential backoff to
        # max_retry_delay_seconds
        current_delay = 0.1

        while True:
            try:
                return await self._executor.submit(func, request)
            except GoogleAPICallError as err:
                message = err.message
                # Raise RuntimeError for exceptions that are not retryable.
                # Otherwise, pass through to retry.
                if err.code not in RETRYABLE_ERROR_CODES:
                    raise EngineException(message) from err

            if current_delay > self.max_retry_delay_seconds:
                raise TimeoutError(f'Reached max retry attempts for error: {message}')
            if self.verbose:
                print(message, file=sys.stderr)
                print(f'Waiting {current_delay} seconds before retrying.', file=sys.stderr)
            await duet.sleep(current_delay)
            current_delay *= 2

    async def create_program_async(
        self,
        project_id: str,
        program_id: Optional[str],
        code: any_pb2.Any,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, quantum.QuantumProgram]:
        """Creates a Quantum Engine program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            code: Properly serialized program code.
            description: An optional description to set on the program.
            labels: Optional set of labels to set on the program.

        Returns:
            Tuple of created program id and program
        """

        parent_name = _project_name(project_id)
        program_name = _program_name_from_ids(project_id, program_id) if program_id else ''
        program = quantum.QuantumProgram(name=program_name, code=code)
        if description:
            program.description = description
        if labels:
            program.labels.update(labels)

        request = quantum.CreateQuantumProgramRequest(parent=parent_name, quantum_program=program)
        program = await self._send_request_async(self.grpc_client.create_quantum_program, request)
        return _ids_from_program_name(program.name)[1], program

    create_program = duet.sync(create_program_async)

    async def get_program_async(
        self, project_id: str, program_id: str, return_code: bool
    ) -> quantum.QuantumProgram:
        """Returns a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            return_code: If True returns the serialized program code.
        """
        request = quantum.GetQuantumProgramRequest(
            name=_program_name_from_ids(project_id, program_id), return_code=return_code
        )
        return await self._send_request_async(self.grpc_client.get_quantum_program, request)

    get_program = duet.sync(get_program_async)

    async def list_programs_async(
        self,
        project_id: str,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ):
        """Returns a list of previously executed quantum programs.

        Args:
            project_id: the id of the project
            created_after: retrieve programs that were created after this date
                or time.
            created_before: retrieve programs that were created after this date
                or time.
            has_labels: retrieve programs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                egardless of the label value will be filtered. For example, to
                uery programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}
        """
        filters = []

        if created_after is not None:
            val = _date_or_time_to_filter_expr('created_after', created_after)
            filters.append(f"create_time >= {val}")
        if created_before is not None:
            val = _date_or_time_to_filter_expr('created_before', created_before)
            filters.append(f"create_time <= {val}")
        if has_labels is not None:
            for k, v in has_labels.items():
                filters.append(f"labels.{k}:{v}")
        request = quantum.ListQuantumProgramsRequest(
            parent=_project_name(project_id), filter=" AND ".join(filters)
        )
        return await self._send_request_async(self.grpc_client.list_quantum_programs, request)

    list_programs = duet.sync(list_programs_async)

    async def set_program_description_async(
        self, project_id: str, program_id: str, description: str
    ) -> quantum.QuantumProgram:
        """Sets the description for a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            description: The new program description.

        Returns:
            The updated quantum program.
        """
        program_resource_name = _program_name_from_ids(project_id, program_id)
        request = quantum.UpdateQuantumProgramRequest(
            name=program_resource_name,
            quantum_program=quantum.QuantumProgram(
                name=program_resource_name, description=description
            ),
            update_mask=field_mask_pb2.FieldMask(paths=['description']),
        )
        return await self._send_request_async(self.grpc_client.update_quantum_program, request)

    set_program_description = duet.sync(set_program_description_async)

    async def _set_program_labels_async(
        self, project_id: str, program_id: str, labels: Dict[str, str], fingerprint: str
    ) -> quantum.QuantumProgram:
        program_resource_name = _program_name_from_ids(project_id, program_id)
        request = quantum.UpdateQuantumProgramRequest(
            name=program_resource_name,
            quantum_program=quantum.QuantumProgram(
                name=program_resource_name, labels=labels, label_fingerprint=fingerprint
            ),
            update_mask=field_mask_pb2.FieldMask(paths=['labels']),
        )
        return await self._send_request_async(self.grpc_client.update_quantum_program, request)

    async def set_program_labels_async(
        self, project_id: str, program_id: str, labels: Dict[str, str]
    ) -> quantum.QuantumProgram:
        """Sets (overwriting) the labels for a previously created quantum
        program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            labels: The entire set of new program labels.

        Returns:
            The updated quantum program.
        """
        program = self.get_program(project_id, program_id, False)
        return await self._set_program_labels_async(
            project_id, program_id, labels, program.label_fingerprint
        )

    set_program_labels = duet.sync(set_program_labels_async)

    async def add_program_labels_async(
        self, project_id: str, program_id: str, labels: Dict[str, str]
    ) -> quantum.QuantumProgram:
        """Adds new labels to a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            labels: New labels to add to the existing program labels.

        Returns:
            The updated quantum program.
        """
        program = await self.get_program_async(project_id, program_id, False)
        old_labels = program.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return await self._set_program_labels_async(
                project_id, program_id, new_labels, fingerprint
            )
        return program

    add_program_labels = duet.sync(add_program_labels_async)

    async def remove_program_labels_async(
        self, project_id: str, program_id: str, label_keys: List[str]
    ) -> quantum.QuantumProgram:
        """Removes labels with given keys from the labels of a previously
        created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            label_keys: Label keys to remove from the existing program labels.

        Returns:
            The updated quantum program.
        """
        program = await self.get_program_async(project_id, program_id, False)
        old_labels = program.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return await self._set_program_labels_async(
                project_id, program_id, new_labels, fingerprint
            )
        return program

    remove_program_labels = duet.sync(remove_program_labels_async)

    async def delete_program_async(
        self, project_id: str, program_id: str, delete_jobs: bool = False
    ) -> None:
        """Deletes a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            delete_jobs: If True will delete all the program's jobs, other this
                will fail if the program contains any jobs.
        """
        request = quantum.DeleteQuantumProgramRequest(
            name=_program_name_from_ids(project_id, program_id), delete_jobs=delete_jobs
        )
        await self._send_request_async(self.grpc_client.delete_quantum_program, request)

    delete_program = duet.sync(delete_program_async)

    @deprecated_parameter(
        deadline='v1.4',
        fix='Use `processor_id` instead of `processor_ids`.',
        parameter_desc='processor_ids',
        match=lambda args, kwargs: _match_deprecated_processor_ids(args, kwargs),
        rewrite=lambda args, kwargs: rewrite_processor_ids_to_processor_id(args, kwargs),
    )
    async def create_job_async(
        self,
        project_id: str,
        program_id: str,
        job_id: Optional[str],
        processor_ids: Optional[Sequence[str]] = None,
        run_context: any_pb2.Any = any_pb2.Any(),
        priority: Optional[int] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        *,
        processor_id: str = "",
        run_name: str = "",
        device_config_name: str = "",
    ) -> Tuple[str, quantum.QuantumJob]:
        """Creates and runs a job on Quantum Engine.

        Either both `run_name` and `device_config_name` must be set, or neither
        of them must be set. If none of them are set, a default internal device
        configuration will be used.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            run_context: Properly serialized run context.
            processor_ids: Deprecated list of candidate processor ids to run the program.
                Only allowed to contain one processor_id. If the argument `processor_id`
                is non-empty, `processor_ids` will be ignored. Otherwise the deprecated
                decorator will fix the arguments and call create_job_async using
                `processor_id` instead of `processor_ids`.
            priority: Optional priority to run at, 0-1000.
            description: Optional description to set on the job.
            labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program. If not set,
                `processor_ids` will be used.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.
        Returns:
            Tuple of created job id and job.

        Raises:
            ValueError: If the priority is not between 0 and 1000.
            ValueError: If neither `processor_id` or `processor_ids` are set.
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
            ValueError: If `processor_ids` has more than one processor id.
            ValueError: If either `run_name` and `device_config_name` are set but
                `processor_id` is empty.
        """
        # Check program to run and program parameters.
        if priority and not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')
        if not processor_id:
            raise ValueError('Must specify a processor id when creating a job.')
        if bool(run_name) ^ bool(device_config_name):
            raise ValueError('Cannot specify only one of `run_name` and `device_config_name`')

        # Create job.
        job_name = _job_name_from_ids(project_id, program_id, job_id) if job_id else ''
        job = quantum.QuantumJob(
            name=job_name,
            scheduling_config=quantum.SchedulingConfig(
                processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                    processor=_processor_name_from_ids(project_id, processor_id),
                    device_config_key=quantum.DeviceConfigKey(
                        run_name=run_name, config_alias=device_config_name
                    ),
                )
            ),
            run_context=run_context,
        )
        if priority:
            job.scheduling_config.priority = priority
        if description:
            job.description = description
        if labels:
            job.labels.update(labels)
        request = quantum.CreateQuantumJobRequest(
            parent=_program_name_from_ids(project_id, program_id), quantum_job=job
        )
        job = await self._send_request_async(self.grpc_client.create_quantum_job, request)
        return _ids_from_job_name(job.name)[2], job

    # TODO(cxing): Remove type ignore once @deprecated_parameter decorator is removed
    create_job = duet.sync(create_job_async)  # type: ignore

    async def list_jobs_async(
        self,
        project_id: str,
        program_id: Optional[str] = None,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.ExecutionStatus.State]] = None,
        executed_processor_ids: Optional[List[str]] = None,
        scheduled_processor_ids: Optional[List[str]] = None,
    ):
        """Returns the list of jobs for a given program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Optional, a unique ID of the program within the parent
                project. If None, jobs will be listed across all programs within
                the project.
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created after this date
                or time.
            has_labels: retrieve jobs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}

            execution_states: retrieve jobs that have an execution state that
                is contained in `execution_states`. See
                `quantum.ExecutionStatus.State` enum for accepted values.

            executed_processor_ids: filters jobs by processor ID used for
                execution. Matches any of provided IDs.
            scheduled_processor_ids: filters jobs by any of provided
                scheduled processor IDs.
        """
        filters = []

        if created_after is not None:
            val = _date_or_time_to_filter_expr('created_after', created_after)
            filters.append(f"create_time >= {val}")
        if created_before is not None:
            val = _date_or_time_to_filter_expr('created_before', created_before)
            filters.append(f"create_time <= {val}")
        if has_labels is not None:
            for k, v in has_labels.items():
                filters.append(f"labels.{k}:{v}")
        if execution_states is not None:
            state_filter = []
            for execution_state in execution_states:
                state_filter.append(f"execution_status.state = {execution_state.name}")
            filters.append(f"({' OR '.join(state_filter)})")
        if executed_processor_ids is not None:
            ids_filter = []
            for processor_id in executed_processor_ids:
                ids_filter.append(f"executed_processor_id = {processor_id}")
            filters.append(f"({' OR '.join(ids_filter)})")
        if scheduled_processor_ids is not None:
            ids_filter = []
            for processor_id in scheduled_processor_ids:
                ids_filter.append(f"scheduled_processor_ids: {processor_id}")
            filters.append(f"({' OR '.join(ids_filter)})")

        if program_id is None:
            program_id = "-"
        parent = _program_name_from_ids(project_id, program_id)
        request = quantum.ListQuantumJobsRequest(parent=parent, filter=" AND ".join(filters))
        return await self._send_request_async(self.grpc_client.list_quantum_jobs, request)

    list_jobs = duet.sync(list_jobs_async)

    async def get_job_async(
        self, project_id: str, program_id: str, job_id: str, return_run_context: bool
    ) -> quantum.QuantumJob:
        """Returns a previously created job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            return_run_context: If true then the run context will be loaded
                from the job's run_context_location and set on the returned
                QuantumJob.
        """
        request = quantum.GetQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id),
            return_run_context=return_run_context,
        )
        return await self._send_request_async(self.grpc_client.get_quantum_job, request)

    get_job = duet.sync(get_job_async)

    async def set_job_description_async(
        self, project_id: str, program_id: str, job_id: str, description: str
    ) -> quantum.QuantumJob:
        """Sets the description for a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            description: The new job description.

        Returns:
            The updated quantum job.
        """
        job_resource_name = _job_name_from_ids(project_id, program_id, job_id)
        request = quantum.UpdateQuantumJobRequest(
            name=job_resource_name,
            quantum_job=quantum.QuantumJob(name=job_resource_name, description=description),
            update_mask=field_mask_pb2.FieldMask(paths=['description']),
        )
        return await self._send_request_async(self.grpc_client.update_quantum_job, request)

    set_job_description = duet.sync(set_job_description_async)

    async def _set_job_labels_async(
        self,
        project_id: str,
        program_id: str,
        job_id: str,
        labels: Dict[str, str],
        fingerprint: str,
    ) -> quantum.QuantumJob:
        job_resource_name = _job_name_from_ids(project_id, program_id, job_id)
        request = quantum.UpdateQuantumJobRequest(
            name=job_resource_name,
            quantum_job=quantum.QuantumJob(
                name=job_resource_name, labels=labels, label_fingerprint=fingerprint
            ),
            update_mask=field_mask_pb2.FieldMask(paths=['labels']),
        )
        return await self._send_request_async(self.grpc_client.update_quantum_job, request)

    async def set_job_labels_async(
        self, project_id: str, program_id: str, job_id: str, labels: Dict[str, str]
    ) -> quantum.QuantumJob:
        """Sets (overwriting) the labels for a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            labels: The entire set of new job labels.

        Returns:
            The updated quantum job.
        """
        job = await self.get_job_async(project_id, program_id, job_id, False)
        return await self._set_job_labels_async(
            project_id, program_id, job_id, labels, job.label_fingerprint
        )

    set_job_labels = duet.sync(set_job_labels_async)

    async def add_job_labels_async(
        self, project_id: str, program_id: str, job_id: str, labels: Dict[str, str]
    ) -> quantum.QuantumJob:
        """Adds new labels to a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            labels: New labels to add to the existing job labels.

        Returns:
            The updated quantum job.
        """
        job = await self.get_job_async(project_id, program_id, job_id, False)
        old_labels = job.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return await self._set_job_labels_async(
                project_id, program_id, job_id, new_labels, fingerprint
            )
        return job

    add_job_labels = duet.sync(add_job_labels_async)

    async def remove_job_labels_async(
        self, project_id: str, program_id: str, job_id: str, label_keys: List[str]
    ) -> quantum.QuantumJob:
        """Removes labels with given keys from the labels of a previously
        created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            label_keys: Label keys to remove from the existing job labels.

        Returns:
            The updated quantum job.
        """
        job = await self.get_job_async(project_id, program_id, job_id, False)
        old_labels = job.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return await self._set_job_labels_async(
                project_id, program_id, job_id, new_labels, fingerprint
            )
        return job

    remove_job_labels = duet.sync(remove_job_labels_async)

    async def delete_job_async(self, project_id: str, program_id: str, job_id: str) -> None:
        """Deletes a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
        """
        request = quantum.DeleteQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id)
        )
        await self._send_request_async(self.grpc_client.delete_quantum_job, request)

    delete_job = duet.sync(delete_job_async)

    async def cancel_job_async(self, project_id: str, program_id: str, job_id: str) -> None:
        """Cancels the given job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
        """
        request = quantum.CancelQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id)
        )
        await self._send_request_async(self.grpc_client.cancel_quantum_job, request)

    cancel_job = duet.sync(cancel_job_async)

    async def get_job_results_async(
        self, project_id: str, program_id: str, job_id: str
    ) -> quantum.QuantumResult:
        """Returns the results of a completed job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.

        Returns:
            The quantum result.
        """
        request = quantum.GetQuantumResultRequest(
            parent=_job_name_from_ids(project_id, program_id, job_id)
        )
        return await self._send_request_async(self.grpc_client.get_quantum_result, request)

    get_job_results = duet.sync(get_job_results_async)

    def run_job_over_stream(
        self,
        *,
        project_id: str,
        program_id: str,
        code: any_pb2.Any,
        run_context: any_pb2.Any,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_id: str,
        priority: Optional[int] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
        processor_id: str = "",
        run_name: str = "",
        device_config_name: str = "",
    ) -> duet.AwaitableFuture[Union[quantum.QuantumResult, quantum.QuantumJob]]:
        """Runs a job with the given program and job information over a stream.

        Sends the request over the Quantum Engine QuantumRunStream bidirectional stream, and returns
        a future for the stream response. The future will be completed with a `QuantumResult` if
        the job is successful; otherwise, it will be completed with a QuantumJob.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            code: Properly serialized program code.
            run_context: Properly serialized run context.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_id: Unique ID of the job within the parent program.
            priority: Optional priority to run at, 0-1000.
            job_description: Optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program. If not set,
                `processor_ids` will be used.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.

        Returns:
            A future for the job result, or the job if the job has failed.

        Raises:
            ValueError: If the priority is not between 0 and 1000.
            ValueError: If `processor_id` is not set.
            ValueError: If only one of `run_name` and `device_config_name` are specified.
        """
        # Check program to run and program parameters.
        if priority and not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')
        if not processor_id:
            raise ValueError('Must specify a processor id when creating a job.')
        if bool(run_name) ^ bool(device_config_name):
            raise ValueError('Cannot specify only one of `run_name` and `device_config_name`')

        project_name = _project_name(project_id)

        program_name = _program_name_from_ids(project_id, program_id)
        program = quantum.QuantumProgram(name=program_name, code=code)
        if program_description:
            program.description = program_description
        if program_labels:
            program.labels.update(program_labels)

        job = quantum.QuantumJob(
            name=_job_name_from_ids(project_id, program_id, job_id),
            scheduling_config=quantum.SchedulingConfig(
                processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                    processor=_processor_name_from_ids(project_id, processor_id),
                    device_config_key=quantum.DeviceConfigKey(
                        run_name=run_name, config_alias=device_config_name
                    ),
                )
            ),
            run_context=run_context,
        )
        if priority:
            job.scheduling_config.priority = priority
        if job_description:
            job.description = job_description
        if job_labels:
            job.labels.update(job_labels)

        return self._stream_manager.submit(project_name, program, job)

    async def list_processors_async(self, project_id: str) -> List[quantum.QuantumProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Args:
            project_id: A project_id of the parent Google Cloud Project.

        Returns:
            A list of metadata of each processor.
        """
        request = quantum.ListQuantumProcessorsRequest(parent=_project_name(project_id), filter='')
        return await self._send_list_request_async(
            self.grpc_client.list_quantum_processors, request
        )

    list_processors = duet.sync(list_processors_async)

    async def get_processor_async(
        self, project_id: str, processor_id: str
    ) -> quantum.QuantumProcessor:
        """Returns a quantum processor.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.

        Returns:
            The quantum processor.
        """
        request = quantum.GetQuantumProcessorRequest(
            name=_processor_name_from_ids(project_id, processor_id)
        )
        return await self._send_request_async(self.grpc_client.get_quantum_processor, request)

    get_processor = duet.sync(get_processor_async)

    async def list_calibrations_async(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> List[quantum.QuantumCalibration]:
        """Returns a list of quantum calibrations.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            filter_str: Filter string current only supports 'timestamp' with values
            of epoch time in seconds or short string 'yyyy-MM-dd'. For example:
                'timestamp > 1577960125 AND timestamp <= 1578241810'
                'timestamp > 2020-01-02 AND timestamp <= 2020-01-05'

        Returns:
            A list of calibrations.
        """
        request = quantum.ListQuantumCalibrationsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        return await self._send_list_request_async(
            self.grpc_client.list_quantum_calibrations, request
        )

    list_calibrations = duet.sync(list_calibrations_async)

    async def get_calibration_async(
        self, project_id: str, processor_id: str, calibration_timestamp_seconds: int
    ) -> quantum.QuantumCalibration:
        """Returns a quantum calibration.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            calibration_timestamp_seconds: The timestamp of the calibration in
                seconds.

        Returns:
            The quantum calibration.
        """
        request = quantum.GetQuantumCalibrationRequest(
            name=_calibration_name_from_ids(project_id, processor_id, calibration_timestamp_seconds)
        )
        return await self._send_request_async(self.grpc_client.get_quantum_calibration, request)

    get_calibration = duet.sync(get_calibration_async)

    async def get_current_calibration_async(
        self, project_id: str, processor_id: str
    ) -> Optional[quantum.QuantumCalibration]:
        """Returns the current quantum calibration for a processor if it has one.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.

        Returns:
            The quantum calibration or None if there is no current calibration.

        Raises:
            EngineException: If the request for calibration fails.
        """
        try:
            request = quantum.GetQuantumCalibrationRequest(
                name=_processor_name_from_ids(project_id, processor_id) + '/calibrations/current'
            )
            return await self._send_request_async(self.grpc_client.get_quantum_calibration, request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    get_current_calibration = duet.sync(get_current_calibration_async)

    async def create_reservation_async(
        self,
        project_id: str,
        processor_id: str,
        start: datetime.datetime,
        end: datetime.datetime,
        whitelisted_users: Optional[List[str]] = None,
    ):
        """Creates a quantum reservation and returns the created object.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
                or None if the engine should generate an id
            start: the starting time of the reservation as a datetime object
            end: the ending time of the reservation as a datetime object
            whitelisted_users: a list of emails that can use the reservation.
        """
        parent = _processor_name_from_ids(project_id, processor_id)
        reservation = quantum.QuantumReservation(
            name='',
            start_time=Timestamp(seconds=int(start.timestamp())),
            end_time=Timestamp(seconds=int(end.timestamp())),
        )
        if whitelisted_users:
            reservation.whitelisted_users.extend(whitelisted_users)
        request = quantum.CreateQuantumReservationRequest(
            parent=parent, quantum_reservation=reservation
        )
        return await self._send_request_async(self.grpc_client.create_quantum_reservation, request)

    create_reservation = duet.sync(create_reservation_async)

    async def cancel_reservation_async(
        self, project_id: str, processor_id: str, reservation_id: str
    ):
        """Cancels a quantum reservation.

        This action is only valid if the associated [QuantumProcessor]
        schedule not been frozen. Otherwise, delete_reservation should
        be used.

        The reservation will be truncated to end at the time when the request is
        serviced and any remaining time will be made available as an open swim
        period. This action will only succeed if the reservation has not yet
        ended and is within the processor's freeze window. If the reservation
        has already ended or is beyond the processor's freeze window, then the
        call will return an error.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
        """
        name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
        request = quantum.CancelQuantumReservationRequest(name=name)
        return await self._send_request_async(self.grpc_client.cancel_quantum_reservation, request)

    cancel_reservation = duet.sync(cancel_reservation_async)

    async def delete_reservation_async(
        self, project_id: str, processor_id: str, reservation_id: str
    ):
        """Deletes a quantum reservation.

        This action is only valid if the associated [QuantumProcessor]
        schedule has not been frozen.  Otherwise, cancel_reservation
        should be used.

        If the reservation has already ended or is within the processor's
        freeze window, then the call will return a `FAILED_PRECONDITION` error.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
        """
        name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
        request = quantum.DeleteQuantumReservationRequest(name=name)
        return await self._send_request_async(self.grpc_client.delete_quantum_reservation, request)

    delete_reservation = duet.sync(delete_reservation_async)

    async def get_reservation_async(
        self, project_id: str, processor_id: str, reservation_id: str
    ) -> Optional[quantum.QuantumReservation]:
        """Gets a quantum reservation from the engine.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project.

        Raises:
            EngineException: If the request to get the reservation failed.
        """
        try:
            name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
            request = quantum.GetQuantumReservationRequest(name=name)
            return await self._send_request_async(self.grpc_client.get_quantum_reservation, request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    get_reservation = duet.sync(get_reservation_async)

    async def list_reservations_async(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> List[quantum.QuantumReservation]:
        """Returns a list of quantum reservations.

        Only reservations owned by this project will be returned.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            filter_str: A string for filtering quantum reservations.
                The fields eligible for filtering are start_time and end_time
                Examples:
                    `start_time >= 1584385200`: Reservation began on or after
                        the epoch time Mar 16th, 7pm GMT.
                    `end_time >= 1483370475`: Reservation ends on
                        or after Jan 2nd 2017 15:21:15

        Returns:
            A list of QuantumReservation objects.
        """
        request = quantum.ListQuantumReservationsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        return await self._send_list_request_async(
            self.grpc_client.list_quantum_reservations, request
        )

    list_reservations = duet.sync(list_reservations_async)

    async def update_reservation_async(
        self,
        project_id: str,
        processor_id: str,
        reservation_id: str,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        whitelisted_users: Optional[List[str]] = None,
    ):
        """Updates a quantum reservation.

        This will update a quantum reservation's starting time, ending time,
        and list of whitelisted users.  If any field is not filled, it will
        not be updated.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
            start: the new starting time of the reservation as a datetime object
            end: the new ending time of the reservation as a datetime object
            whitelisted_users: a list of emails that can use the reservation.
                The empty list, [], will clear the whitelisted_users while None
                will leave the value unchanged.
        """
        name = (
            _reservation_name_from_ids(project_id, processor_id, reservation_id)
            if reservation_id
            else ''
        )

        reservation = quantum.QuantumReservation(name=name)
        paths = []
        if start:
            reservation.start_time = start
            paths.append('start_time')
        if end:
            reservation.end_time = end
            paths.append('end_time')
        if whitelisted_users is not None:
            reservation.whitelisted_users.extend(whitelisted_users)
            paths.append('whitelisted_users')

        request = quantum.UpdateQuantumReservationRequest(
            name=name,
            quantum_reservation=reservation,
            update_mask=field_mask_pb2.FieldMask(paths=paths),
        )
        return await self._send_request_async(self.grpc_client.update_quantum_reservation, request)

    update_reservation = duet.sync(update_reservation_async)

    async def list_time_slots_async(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> List[quantum.QuantumTimeSlot]:
        """Returns a list of quantum time slots on a processor.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            filter_str:  A string expression for filtering the quantum
                time slots returned by the list command. The fields
                eligible for filtering are `start_time`, `end_time`.

        Returns:
            A list of QuantumTimeSlot objects.
        """
        request = quantum.ListQuantumTimeSlotsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        return await self._send_list_request_async(
            self.grpc_client.list_quantum_time_slots, request
        )

    list_time_slots = duet.sync(list_time_slots_async)


def _project_name(project_id: str) -> str:
    return f'projects/{project_id}'


def _program_name_from_ids(project_id: str, program_id: str) -> str:
    return f'projects/{project_id}/programs/{program_id}'


def _job_name_from_ids(project_id: str, program_id: str, job_id: str) -> str:
    return f'projects/{project_id}/programs/{program_id}/jobs/{job_id}'


def _processor_name_from_ids(project_id: str, processor_id: str) -> str:
    return f'projects/{project_id}/processors/{processor_id}'


def _calibration_name_from_ids(
    project_id: str, processor_id: str, calibration_time_seconds: int
) -> str:
    return (
        f'projects/{project_id}/processors/{processor_id}/calibrations/{calibration_time_seconds}'
    )


def _reservation_name_from_ids(project_id: str, processor_id: str, reservation_id: str) -> str:
    return f'projects/{project_id}/processors/{processor_id}/reservations/{reservation_id}'


def _ids_from_program_name(program_name: str) -> Tuple[str, str]:
    parts = program_name.split('/')
    return parts[1], parts[3]


def _ids_from_job_name(job_name: str) -> Tuple[str, str, str]:
    parts = job_name.split('/')
    return parts[1], parts[3], parts[5]


def _ids_from_processor_name(processor_name: str) -> Tuple[str, str]:
    parts = processor_name.split('/')
    return parts[1], parts[3]


def _ids_from_calibration_name(calibration_name: str) -> Tuple[str, str, int]:
    parts = calibration_name.split('/')
    return parts[1], parts[3], int(parts[5])


def _date_or_time_to_filter_expr(param_name: str, param: Union[datetime.datetime, datetime.date]):
    """Formats datetime or date to filter expressions.

    Args:
        param_name: The name of the filter parameter (for error messaging).
        param: The value of the parameter.

    Raises:
        ValueError: If the supplied param is not a datetime or date.
    """
    if isinstance(param, datetime.datetime):
        return f"{int(param.timestamp())}"
    elif isinstance(param, datetime.date):
        return f"{param.isoformat()}"

    raise ValueError(
        f"Unsupported date/time type for {param_name}: got {param} of "
        f"type {type(param)}. Supported types: datetime.datetime and"
        f"datetime.date"
    )


def rewrite_processor_ids_to_processor_id(args, kwargs):
    """Rewrites the create_job parameters so that `processor_id` is used instead of the deprecated
    `processor_ids`.

        Raises:
            ValueError: If `processor_ids` has more than one processor id.
            ValueError: If `run_name` or `device_config_name` are set but `processor_id` is not.
    """

    # Use `processor_id` keyword argument instead of `processor_ids`
    processor_ids = args[4] if len(args) > 4 else kwargs['processor_ids']
    if len(processor_ids) > 1:
        raise ValueError("The use of multiple processors is no longer supported.")
    if 'processor_id' not in kwargs or not kwargs['processor_id']:
        if ('run_name' in kwargs and kwargs['run_name']) or (
            'device_config_name' in kwargs and kwargs['device_config_name']
        ):
            raise ValueError(
                'Cannot specify `run_name` or `device_config_name` if `processor_id` is empty.'
            )
        kwargs['processor_id'] = processor_ids[0]

    # Erase `processor_ids` from args and kwargs
    if len(args) > 4:
        args_list = list(args)
        args_list[4] = None
        args = tuple(args_list)
    else:
        kwargs.pop('processor_ids')

    return args, kwargs


def _match_deprecated_processor_ids(args, kwargs):
    return ('processor_ids' in kwargs and kwargs['processor_ids']) or len(args) > 4

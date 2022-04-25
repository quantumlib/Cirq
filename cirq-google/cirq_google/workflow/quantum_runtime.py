# Copyright 2021 The Cirq Developers
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

"""Runtime information dataclasses and execution of executables."""
import abc
import contextlib
import dataclasses
import queue
import threading
import datetime
import time
import uuid
from typing import Any, Dict, Optional, List, TYPE_CHECKING, Union

import numpy as np

import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow.io import _FilesystemSaver
from cirq_google.workflow.progress import _PrintLogger
from cirq_google.workflow.quantum_executable import (
    QuantumExecutable,
    ExecutableSpec,
    QuantumExecutableGroup,
)
from cirq_google.workflow.qubit_placement import QubitPlacer, NaiveQubitPlacer

if TYPE_CHECKING:
    import cirq_google as cg


@dataclasses.dataclass
class SharedRuntimeInfo:
    """Runtime information common to all `cg.QuantumExecutable`s in an execution of a
    `cg.QuantumExecutableGroup`.

    There is one `cg.SharedRuntimeInfo` per `cg.ExecutableGroupResult`.

    Args:
        run_id: A unique `str` identifier for this run.
        device: The actual device used during execution, not just its processor_id
    """

    run_id: str
    device: Optional[cirq.Device] = None
    run_start_time: Optional[datetime.datetime] = None
    run_end_time: Optional[datetime.datetime] = None

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


def _try_tuple(k: Any) -> Any:
    """If we serialize a dictionary that had tuple keys, they get turned to json lists."""
    if isinstance(k, list):
        return tuple(k)
    return k  # coverage: ignore


@dataclasses.dataclass
class RuntimeInfo:
    """Runtime information relevant to a particular `cg.QuantumExecutable`.

    There is one `cg.RuntimeInfo` per `cg.ExecutableResult`

    Args:
        execution_index: What order (in its `cg.QuantumExecutableGroup`) this
            `cg.QuantumExecutable` was executed.
        qubit_placement: If a QubitPlacer was used, a record of the mapping
            from problem-qubits to device-qubits.
        timings_s: The durations of measured subroutines. Each entry in this
            dictionary maps subroutine name to the amount of time the subroutine
            took in units of seconds.
    """

    execution_index: int
    qubit_placement: Optional[Dict[Any, cirq.Qid]] = None
    timings_s: Dict[str, float] = dataclasses.field(default_factory=dict)

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        d = dataclass_json_dict(self)
        if d['qubit_placement']:
            d['qubit_placement'] = list(d['qubit_placement'].items())
        d['timings_s'] = list(d['timings_s'].items())
        return d

    @classmethod
    def _from_json_dict_(cls, **kwargs) -> 'RuntimeInfo':
        kwargs.pop('cirq_type')
        if kwargs.get('qubit_placement', None):
            kwargs['qubit_placement'] = {_try_tuple(k): v for k, v in kwargs['qubit_placement']}
        if 'timings_s' in kwargs:
            kwargs['timings_s'] = {k: v for k, v in kwargs['timings_s']}
        return cls(**kwargs)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class ExecutableResult:
    """Results for a `cg.QuantumExecutable`.

    Args:
        spec: The `cg.ExecutableSpec` typifying the `cg.QuantumExecutable`.
        runtime_info: A `cg.RuntimeInfo` dataclass containing information gathered during
            execution of the `cg.QuantumExecutable`.
        raw_data: The `cirq.Result` containing the data from the run.
    """

    spec: Optional[ExecutableSpec]
    runtime_info: RuntimeInfo
    raw_data: cirq.Result

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class ExecutableGroupResult:
    """Results for a `cg.QuantumExecutableGroup`.

    Args:
        runtime_configuration: The `cg.QuantumRuntimeConfiguration` describing how the
            `cg.QuantumExecutableGroup` was requested to be executed.
        shared_runtime_info: A `cg.SharedRuntimeInfo` dataclass containing information gathered
            during execution of the `cg.QuantumExecutableGroup` which is relevant to all
            `executable_results`.
        executable_results: A list of `cg.ExecutableResult`. Each contains results and raw data
            for an individual `cg.QuantumExecutable`.
    """

    runtime_configuration: 'QuantumRuntimeConfiguration'
    shared_runtime_info: SharedRuntimeInfo
    executable_results: List[ExecutableResult]

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class QuantumRuntimeConfiguration:
    """User-requested configuration of how to execute a given `cg.QuantumExecutableGroup`.

    Args:
        processor: The `cg.AbstractEngineProcessor` responsible for running circuits and providing
            device information.
        run_id: A unique `str` identifier for a run. If data already exists for the specified
            `run_id`, an exception will be raised. If not specified, we will generate a UUID4
            run identifier.
        random_seed: An initial seed to make the run deterministic. Otherwise, the default numpy
            seed will be used.
        qubit_placer: A `cg.QubitPlacer` implementation to map executable qubits to device qubits.
            The placer is only called if a given `cg.QuantumExecutable` has a `problem_topology`.
            This subroutine's runtime is keyed by "placement" in `RuntimeInfo.timings_s`.
        target_gateset: If not `None`, compile all circuits to this target gateset prior to
            execution with `cirq.optimize_for_target_gateset`.
    """

    processor_record: 'cg.ProcessorRecord'
    run_id: Optional[str] = None
    random_seed: Optional[int] = None
    qubit_placer: QubitPlacer = NaiveQubitPlacer()
    target_gateset: Optional[cirq.CompilationTargetGateset] = None

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@contextlib.contextmanager
def _time_into_runtime_info(runtime_info: RuntimeInfo, name: str):
    """A context manager that appends timing information into a cg.RuntimeInfo.

    Timings are reported in fractional seconds as reported by `time.monotonic()`.

    Args:
        runtime_info: The runtime information object whose `.timings_s` dictionary will be updated.
        name: A string key name to use in the dictionary.
    """
    start = time.monotonic()
    yield
    runtime_info.timings_s[name] = time.monotonic() - start


class _JobSubmitter(metaclass=abc.ABCMeta):
    """A Queue-based submission system.

    Descendants are callable and suitable for launching via a daemon Thread.

    Descendants use two queues: an input and output queue. The former is filled with
    _InputQueueT objects which contain all information required to call
    `cg.engine.AbstractProcessor.run` and its friends. The `collect()` method is responsible
    for getting these tasks from the input queue. It will mark them as done and then push
    `_JobSubmissionResult` objects to the output queue for final processing and bookkeeping
    by the main thread.
    """

    def __init__(self):
        self._input_q = None
        self._output_q = None
        self._sampler = None

    def set_up_plumbing(self, input_q: queue.Queue, output_q: queue.Queue, sampler: 'cirq.Sampler'):
        self._input_q = input_q
        self._output_q = output_q
        self._sampler = sampler

    def collect(self) -> None:
        """Pull from queue etc."""

    def __call__(self):
        while True:
            self.collect()


@dataclasses.dataclass(frozen=True)
class _FlushJobSubmission:
    """A sentinel class that can be enqueued to signal that the consumer should flush its queue."""


@dataclasses.dataclass(frozen=True)
class _JobSubmissionRequest:
    """An internal-use dataclass for submitting to _JobSubmitter input queues."""

    circuit: cirq.FrozenCircuit
    n_reps: int
    runtime_info: RuntimeInfo


@dataclasses.dataclass(frozen=True)
class _JobSubmissionResult:
    """An internal-use dataclass returned by _JobSubmitter output queues."""

    result: cirq.Result
    runtime_info: RuntimeInfo


_InputQueueT = Union[_FlushJobSubmission, _JobSubmissionRequest]


class _SerialJobSubmitter(_JobSubmitter):
    def collect(self):
        request: _InputQueueT = self._input_q.get()
        if isinstance(request, _FlushJobSubmission):
            # The serial job submitter ignores flush commands, as submission is serial.
            self._input_q.task_done()
            return

        print('Processing', request.runtime_info.execution_index)
        with _time_into_runtime_info(request.runtime_info, 'run'):
            sampler_run_result = self._sampler.run(request.circuit, repetitions=request.n_reps)

        self._output_q.put(
            _JobSubmissionResult(result=sampler_run_result, runtime_info=request.runtime_info)
        )
        self._input_q.task_done()


class _BatchingJobSubmitter(_JobSubmitter):
    def __init__(
        self,
        batch_size: int = 100,
    ):
        super().__init__()
        self.batch_size = batch_size

    def collect(self):
        requests: List[_JobSubmissionRequest] = []
        flush = False
        while len(requests) < self.batch_size:
            request: _InputQueueT = self._input_q.get()
            if isinstance(request, _FlushJobSubmission):
                flush = True
                break

            requests.append(request)

        if len(requests) == 0 and flush:
            # If we have nothing but flush
            self._input_q.task_done()
            return

        (n_reps,) = set(req.n_reps for req in requests)
        print('Running', [req.runtime_info.execution_index for req in requests])

        start_s = time.monotonic()
        nested_results = self._sampler.run_batch(
            [req.circuit for req in requests], repetitions=n_reps
        )
        cirq_results = [sweep_res[0] for sweep_res in nested_results]
        end_s = time.monotonic()

        rt_infos = [req.runtime_info for req in requests]
        assert len(rt_infos) == len(cirq_results)
        for cirq_res, rt_info in zip(cirq_results, rt_infos):
            rt_info.timings_s['run'] = end_s - start_s
            self._output_q.put(_JobSubmissionResult(result=cirq_res, runtime_info=rt_info))
            self._input_q.task_done()

        if flush:
            # Make sure we mark the flush "task" as done.
            # Please note that we do not forward the flush task to the `output_q`
            self._input_q.task_done()


def execute(
    rt_config: QuantumRuntimeConfiguration,
    executable_group: QuantumExecutableGroup,
    base_data_dir: str = ".",
) -> ExecutableGroupResult:
    """Execute a `cg.QuantumExecutableGroup` according to a `cg.QuantumRuntimeConfiguration`.

    The ExecutableGroupResult's constituent parts will be saved to disk as they become
    available. Within the "{base_data_dir}/{run_id}" directory we save:
        - The `cg.QuantumRuntimeConfiguration` at the start of the execution as a record
          of *how* the executable group was run.
        - A `cg.SharedRuntimeInfo` which is updated throughout the run.
        - An `cg.ExecutableResult` for each `cg.QuantumExecutable` as they become available.
        - A `cg.ExecutableGroupResultFilesystemRecord` which is updated throughout the run.

    Args:
        rt_config: The `cg.QuantumRuntimeConfiguration` specifying how to execute
            `executable_group`.
        executable_group: The `cg.QuantumExecutableGroup` containing the executables to execute.
        base_data_dir: Each data file will be written to the "{base_data_dir}/{run_id}/" directory,
            which must not already exist.

    Returns:
        The `cg.ExecutableGroupResult` containing all data and metadata for an execution.

    Raises:
        NotImplementedError: If an executable uses the `params` field or anything other than
            a BitstringsMeasurement measurement field.
        ValueError: If `base_data_dir` is not a valid directory.
    """
    # run_id defaults logic.
    if rt_config.run_id is None:
        run_id = str(uuid.uuid4())
    else:
        run_id = rt_config.run_id

    # base_data_dir handling.
    if not base_data_dir:
        # coverage: ignore
        raise ValueError("Please provide a non-empty `base_data_dir`.")

    sampler = rt_config.processor_record.get_sampler()
    device = rt_config.processor_record.get_device()

    shared_rt_info = SharedRuntimeInfo(
        run_id=run_id, device=device, run_start_time=datetime.datetime.now(tz=datetime.timezone.utc)
    )
    executable_results = []

    saver = _FilesystemSaver(base_data_dir=base_data_dir, run_id=run_id)
    saver.initialize(rt_config, shared_rt_info)

    logger = _PrintLogger(n_total=len(executable_group))
    logger.initialize()

    input_q = queue.Queue()
    output_q = queue.Queue()

    # TODO: let outsiders specify this? as part of runtime_configuration?
    submitter: _JobSubmitter = _SerialJobSubmitter()
    submitter: _JobSubmitter = _BatchingJobSubmitter(batch_size=100)
    # -------------------------

    submitter.set_up_plumbing(input_q=input_q, output_q=output_q, sampler=sampler)
    threading.Thread(target=submitter, daemon=True).start()
    rs = np.random.RandomState(rt_config.random_seed)
    exe: QuantumExecutable
    for i, exe in enumerate(executable_group):
        runtime_info = RuntimeInfo(execution_index=i)

        if exe.params != tuple():
            raise NotImplementedError("Circuit params are not yet supported.")
        if not hasattr(exe.measurement, 'n_repetitions'):
            raise NotImplementedError("Only `BitstringsMeasurement` are supported.")

        circuit = exe.circuit
        if exe.problem_topology is not None:
            with _time_into_runtime_info(runtime_info, 'placement'):
                circuit, mapping = rt_config.qubit_placer.place_circuit(
                    circuit,
                    problem_topology=exe.problem_topology,
                    shared_rt_info=shared_rt_info,
                    rs=rs,
                )
                runtime_info.qubit_placement = mapping
        if rt_config.target_gateset is not None:
            circuit = cirq.optimize_for_target_gateset(
                circuit, gateset=rt_config.target_gateset
            ).freeze()

        input_q.put(
            _JobSubmissionRequest(
                circuit=circuit, n_reps=exe.measurement.n_repetitions, runtime_info=runtime_info
            )
        )

    input_q.put(_FlushJobSubmission())

    for i, exe in enumerate(executable_group):
        submission_result: _JobSubmissionResult = output_q.get()
        exe_result = ExecutableResult(
            spec=exe.spec,
            runtime_info=submission_result.runtime_info,
            raw_data=submission_result.result,
        )
        # Do bookkeeping for finished ExecutableResult
        executable_results.append(exe_result)
        saver.consume_result(exe_result, shared_rt_info)
        logger.consume_result(exe_result, shared_rt_info)
        output_q.task_done()

    input_q.join()
    output_q.join()
    shared_rt_info.run_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    saver.finalize(shared_rt_info=shared_rt_info)
    logger.finalize()

    return ExecutableGroupResult(
        runtime_configuration=rt_config,
        shared_runtime_info=shared_rt_info,
        executable_results=executable_results,
    )

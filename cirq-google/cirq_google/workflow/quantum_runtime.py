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
import contextlib
import dataclasses
import time
import uuid
from typing import Any, Dict, Optional, List, TYPE_CHECKING

import cirq
import numpy as np
from cirq import _compat
from cirq.protocols import dataclass_json_dict, obj_to_dict_helper
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

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        # TODO (gh-4699): serialize `device` as well once SerializableDevice is serializable.
        return obj_to_dict_helper(self, attribute_names=['run_id'])

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
    """

    processor_record: 'cg.ProcessorRecord'
    run_id: Optional[str] = None
    random_seed: Optional[int] = None
    qubit_placer: QubitPlacer = NaiveQubitPlacer()

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
        run_id=run_id,
        device=device,
    )
    executable_results = []

    saver = _FilesystemSaver(base_data_dir=base_data_dir, run_id=run_id)
    saver.initialize(rt_config, shared_rt_info)

    logger = _PrintLogger(n_total=len(executable_group))
    logger.initialize()

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

        with _time_into_runtime_info(runtime_info, 'run'):
            sampler_run_result = sampler.run(circuit, repetitions=exe.measurement.n_repetitions)

        exe_result = ExecutableResult(
            spec=exe.spec,
            runtime_info=runtime_info,
            raw_data=sampler_run_result,
        )
        # Do bookkeeping for finished ExecutableResult
        executable_results.append(exe_result)
        saver.consume_result(exe_result, shared_rt_info)
        logger.consume_result(exe_result, shared_rt_info)

    saver.finalize()
    logger.finalize()

    return ExecutableGroupResult(
        runtime_configuration=rt_config,
        shared_runtime_info=shared_rt_info,
        executable_results=executable_results,
    )

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
from typing import List, Optional

from pyquil import get_qc
from pyquil.api import QuantumComputer
from qcs_sdk.client import QCSClient
from qcs_sdk.qpu import list_quantum_processors
from qcs_sdk.qpu.isa import get_instruction_set_architecture, InstructionSetArchitecture
from qcs_sdk.qpu.translation import get_quilt_calibrations

import cirq
from cirq_rigetti import circuit_sweep_executors as executors, circuit_transformers as transformers
from cirq_rigetti.deprecation import deprecated_cirq_rigetti_class, deprecated_cirq_rigetti_function
from cirq_rigetti.sampler import RigettiQCSSampler

_default_executor = executors.with_quilc_compilation_and_cirq_parameter_resolution


@deprecated_cirq_rigetti_class()
class RigettiQCSService:
    """This class supports running circuits on QCS quantum hardware as well as
    pyQuil's quantum virtual machine (QVM). When sampling a parametric circuit
    across a parameter sweep, use `RigettiQCSSampler` instead.

    This class also includes a number of convenience static methods for describing
    Rigetti quantum processors through the `qcs_api_client`."""

    def __init__(
        self,
        quantum_computer: QuantumComputer,
        executor: executors.CircuitSweepExecutor = _default_executor,
        transformer: transformers.CircuitTransformer = transformers.default,
    ):
        """Initializes a `RigettiQCSService`.

        Args:
            quantum_computer: A pyquil.api.QuantumComputer against which to run the `cirq.Circuit`s.
            executor: A callable that first uses the below transformer on cirq.Circuit s and
                then executes the transformed circuit on the quantum_computer. You may pass your
                own callable or any static method on CircuitSweepExecutors.
            transformer: A callable that transforms the cirq.Circuit into a pyquil.Program.
                You may pass your own callable or any static method on CircuitTransformers.
        """
        self._quantum_computer = quantum_computer
        self._executor = executor
        self._transformer = transformer

    def run(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
    ) -> cirq.Result:
        """Run the given circuit on the QuantumComputer with which the user initialized the service.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            param_resolver: A cirq.ParamResolver to resolve parameters in  circuit.

        Returns:
            A cirq.Result.
        """

        results = self._executor(
            quantum_computer=self._quantum_computer,
            circuit=circuit,
            resolvers=[param_resolver],
            repetitions=repetitions,
            transformer=self._transformer,
        )
        assert len(results) == 1
        return results[0]

    def sampler(self) -> RigettiQCSSampler:
        """Initializes a cirq.Sampler object for invoking the sampler interface.

        Returns:
            A cirq.Sampler for running on the requested quantum_computer.
        """
        return RigettiQCSSampler(
            quantum_computer=self._quantum_computer,
            executor=self._executor,
            transformer=self._transformer,
        )

    @staticmethod
    def list_quantum_processors(client: Optional[QCSClient] = None) -> List[str]:
        """Retrieve a list of available Rigetti quantum processors.

        Args:
            client: Optional; A `QCSClient` initialized with Rigetti QCS credentials
            and configuration. If not provided, `qcs_api_client` will initialize a
            configured client based on configured values in the current user's
            `~/.qcs` directory or default values.

        Returns:
            A qcs_api_client.models.ListQuantumProcessorsResponse containing the identifiers
            of the available quantum processors..
        """
        return list_quantum_processors(client=client)

    @staticmethod
    def get_quilt_calibrations(
        quantum_processor_id: str, client: Optional[QCSClient] = None
    ) -> str:  # pragma: no cover
        """Retrieve the calibration data used for client-side Quil-T generation.

        Args:
            quantum_processor_id: The identifier of the Rigetti QCS quantum processor.
            client: Optional; A `QCSClient` initialized with Rigetti QCS credentials
            and configuration. If not provided, `qcs_api_client` will initialize a
            configured client based on configured values in the current user's
            `~/.qcs` directory or default values.

        Returns:
            A qcs_api_client.models.GetQuiltCalibrationsResponse containing the
            device calibrations.
        """
        return get_quilt_calibrations(quantum_processor_id=quantum_processor_id, client=client)

    @staticmethod
    def get_instruction_set_architecture(
        quantum_processor_id: str, client: Optional[QCSClient] = None
    ) -> InstructionSetArchitecture:  # pragma: no cover
        """Retrieve the Instruction Set Architecture of a QuantumProcessor by ID. This
        includes site specific operations and native gate capabilities.

        Args:
            quantum_processor_id: The identifier of the Rigetti QCS quantum processor.
            client: Optional; A `QCSClient` initialized with Rigetti QCS credentials
            and configuration. If not provided, `qcs_api_client` will initialize a
            configured client based on configured values in the current user's
            `~/.qcs` directory or default values.

        Returns:
            A qcs_api_client.models.InstructionSetArchitecture containing the device specification.
        """
        return get_instruction_set_architecture(
            quantum_processor_id=quantum_processor_id, client=client
        )


@deprecated_cirq_rigetti_function()
def get_rigetti_qcs_service(
    quantum_processor_id: str,
    *,
    as_qvm: Optional[bool] = None,
    noisy: Optional[bool] = None,
    executor: executors.CircuitSweepExecutor = _default_executor,
    transformer: transformers.CircuitTransformer = transformers.default,
) -> RigettiQCSService:
    """Calls `pyquil.get_qc` to initialize a `pyquil.api.QuantumComputer` and uses
    this to initialize `RigettiQCSService`.

    Args:
        quantum_processor_id: The name of the desired quantum computer. This should
            correspond to a name returned by `pyquil.api.list_quantum_computers`. Names
            ending in "-qvm" will return a QVM. Names ending in "-pyqvm" will return a
            `pyquil.PyQVM`. Otherwise, we will return a Rigetti QCS QPU if one exists
            with the requested name.
        as_qvm: An optional flag to force construction of a QVM (instead of a QPU). If
            specified and set to `True`, a QVM-backed quantum computer will be returned regardless
            of the name's suffix
        noisy: An optional flag to force inclusion of a noise model. If
            specified and set to `True`, a quantum computer with a noise model will be returned.
            The generic QVM noise model is simple T1 and T2 noise plus readout error. At the time
            of this writing, this has no effect on a QVM initialized based on a Rigetti QCS
            `qcs_api_client.models.InstructionSetArchitecture`.
        executor: A callable that first uses the below transformer on cirq.Circuit s and
            then executes the transformed circuit on the quantum_computer. You may pass your
            own callable or any static method on CircuitSweepExecutors.
        transformer: A callable that transforms the cirq.Circuit into a pyquil.Program.
            You may pass your own callable or any static method on CircuitTransformers.

    Returns:
        A `RigettiQCSService` with the specified quantum processor, executor, and transformer.

    """
    qc = get_qc(quantum_processor_id, as_qvm=as_qvm, noisy=noisy)
    return RigettiQCSService(quantum_computer=qc, executor=executor, transformer=transformer)

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

from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    Optional,
    List,
    Union,
    Generic,
    TypeVar,
    Dict,
)

from unittest.mock import create_autospec, Mock
import pytest
from pyquil import Program
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.api import (
    QAM,
    QuantumComputer,
    QuantumExecutable,
    QAMExecutionResult,
    EncryptedProgram,
    MemoryMap,
)
from pyquil.api._abstract_compiler import AbstractCompiler
from qcs_sdk import QCSClient, ExecutionData, ResultData, RegisterData
from qcs_sdk.qvm import QVMResultData
import networkx as nx
import cirq
import sympy
import numpy as np

T = TypeVar("T")


class MockQAM(QAM, Generic[T]):
    _run_count: int
    _mock_results: Dict[str, np.ndarray]

    def __init__(self, *args, **kwargs) -> None:
        self._run_count = 0
        self._mock_results: Dict[str, np.ndarray] = {}

    def execute(
        self,
        executable: Union[EncryptedProgram, Program],
        memory_map: Optional[Mapping[str, Union[Sequence[int], Sequence[float]]]] = ...,
        **kwargs: Any,
    ) -> Any:
        pass

    def execute_with_memory_map_batch(  # type: ignore[empty-body]
        self, executable: QuantumExecutable, memory_maps: Iterable[MemoryMap], **kwargs: Any
    ) -> List[T]:
        pass

    def run(
        self,
        executable: Union[EncryptedProgram, Program],
        memory_map: Optional[Mapping[str, Union[Sequence[int], Sequence[float]]]] = ...,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def get_result(self, execute_response: T) -> QAMExecutionResult:
        raise NotImplementedError


class MockCompiler(AbstractCompiler):
    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool] = None) -> Program:
        raise NotImplementedError

    def native_quil_to_executable(self, nq_program: Program, **kwargs: Any) -> QuantumExecutable:
        raise NotImplementedError

    def reset(self) -> None:
        pass


@pytest.fixture
def qam() -> QAM:
    return MockQAM()


@pytest.fixture
def quantum_processor() -> AbstractQuantumProcessor:
    topology = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    return NxQuantumProcessor(topology=topology)


@pytest.fixture
def qcs_client() -> QCSClient:
    return QCSClient()


@pytest.fixture
def compiler(quantum_processor, qcs_client) -> AbstractCompiler:
    return MockCompiler(
        client_configuration=qcs_client, timeout=0, quantum_processor=quantum_processor
    )


@pytest.fixture
def quantum_computer(qam: QAM, compiler: AbstractCompiler) -> QuantumComputer:
    return QuantumComputer(name='mocked', qam=qam, compiler=compiler)


@pytest.fixture
def bell_circuit_with_qids() -> Tuple[cirq.Circuit, List[cirq.LineQubit]]:
    bell_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    bell_circuit.append(cirq.H(qubits[0]))
    bell_circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    bell_circuit.append(cirq.measure(qubits[0], qubits[1], key='m'))

    return bell_circuit, qubits


@pytest.fixture
def bell_circuit(bell_circuit_with_qids: Tuple[cirq.Circuit, List[cirq.LineQubit]]) -> cirq.Circuit:
    return bell_circuit_with_qids[0]


@pytest.fixture
def parametric_circuit_with_params() -> Tuple[cirq.Circuit, cirq.Linspace]:
    q = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))

    # Sweep exponent from zero (off) to one (on) and back to two (off)
    param_sweep = cirq.Linspace('t', start=0, stop=2, length=5)

    return circuit, param_sweep


class MockQPUImplementer:
    def __init__(self, quantum_computer: QuantumComputer):
        """Initializes a MockQPUImplementer.

        Args:
            quantum_computer: QuantumComputer to mock.
        """
        self.quantum_computer = quantum_computer

    def implement_passive_quantum_computer_with_results(
        self, results: List[np.ndarray]
    ) -> QuantumComputer:
        """Mocks compilation methods on the `quantum_computer.compiler`, passively passing the
        `Program` through. Sequentially adds results to the
        `quantum_computer.qam._memory_region` (this will not work for asynchronous runs).

        Args:
            results: np.ndarray to sequentially write to `QAM._memory_region`.

        Returns:
            A mocked QuantumComputer.
        """
        quantum_computer = self.quantum_computer

        def quil_to_native_quil(program: Program, *, protoquil: Optional[bool] = None) -> Program:
            return program

        quantum_computer.compiler.quil_to_native_quil = create_autospec(  # type: ignore
            quantum_computer.compiler.quil_to_native_quil, side_effect=quil_to_native_quil
        )

        def native_quil_to_executable(nq_program: Program) -> QuantumExecutable:
            assert 2 == nq_program.num_shots
            return nq_program

        quantum_computer.compiler.native_quil_to_executable = create_autospec(  # type: ignore
            quantum_computer.compiler.native_quil_to_executable,
            side_effect=native_quil_to_executable,
        )

        def run(
            program: Union[Program, EncryptedProgram], memory_map: MemoryMap
        ) -> QAMExecutionResult:
            qam = quantum_computer.qam
            qam._mock_results = qam._mock_results or {}  # type: ignore
            qam._mock_results["m0"] = results[qam._run_count]  # type: ignore

            quantum_computer.qam._run_count += 1  # type: ignore
            return QAMExecutionResult(
                executable=program,
                data=ExecutionData(
                    result_data=ResultData.from_qvm(
                        QVMResultData.from_memory_map(
                            {
                                k: RegisterData.from_f64([v])
                                for k, v in qam._mock_results.items()  # type: ignore
                            }
                        )
                    )
                ),
            )

        quantum_computer.qam.run = Mock(quantum_computer.qam.run, side_effect=run)  # type: ignore
        return quantum_computer


@pytest.fixture
def mock_qpu_implementer(quantum_computer) -> MockQPUImplementer:
    return MockQPUImplementer(quantum_computer=quantum_computer)

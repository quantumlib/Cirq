# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import Any, List, Sequence, Tuple
import cirq
import pytest
from pyquil import Program
from pyquil.api import QuantumComputer
import numpy as np
from pyquil.gates import MEASURE, RX, X, DECLARE, H, CNOT
from cirq_rigetti import RigettiQCSService
from typing_extensions import Protocol
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors


_default_executor = executors.with_quilc_compilation_and_cirq_parameter_resolution


class _ResultBuilder(Protocol):
    def __call__(
        self,
        mock_qpu_implementer: Any,
        circuit: cirq.Circuit,
        sweepable: cirq.Sweepable,
        *,
        executor: executors.CircuitSweepExecutor = _default_executor,
        transformer: transformers.CircuitTransformer = transformers.default,
    ) -> Tuple[Sequence[cirq.Result], QuantumComputer, List[np.ndarray], List[cirq.ParamResolver]]:
        pass


def _build_service_results(
    mock_qpu_implementer: Any,
    circuit: cirq.Circuit,
    sweepable: cirq.Sweepable,
    *,
    executor: executors.CircuitSweepExecutor = _default_executor,
    transformer: transformers.CircuitTransformer = transformers.default,
) -> Tuple[Sequence[cirq.Result], QuantumComputer, List[np.ndarray], List[cirq.ParamResolver]]:
    repetitions = 2
    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    param_resolver_index = min(1, len(param_resolvers) - 1)
    param_resolver = param_resolvers[param_resolver_index]

    expected_results = [
        np.ones((repetitions,))
        * (param_resolver["t"] if "t" in param_resolver else param_resolver_index)
    ]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(
        expected_results
    )
    service = RigettiQCSService(
        quantum_computer=quantum_computer, executor=executor, transformer=transformer
    )

    result = service.run(circuit=circuit, param_resolver=param_resolver, repetitions=repetitions)
    return [result], quantum_computer, expected_results, [param_resolver]


def _build_sampler_results(
    mock_qpu_implementer: Any,
    circuit: cirq.Circuit,
    sweepable: cirq.Sweepable,
    *,
    executor: executors.CircuitSweepExecutor = _default_executor,
    transformer: transformers.CircuitTransformer = transformers.default,
) -> Tuple[Sequence[cirq.Result], QuantumComputer, List[np.ndarray], cirq.Sweepable]:
    repetitions = 2

    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [
        np.ones((repetitions,)) * (params["t"] if "t" in params else i)
        for i, params in enumerate(param_resolvers)
    ]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(
        expected_results
    )
    service = RigettiQCSService(
        quantum_computer=quantum_computer, executor=executor, transformer=transformer
    )

    sampler = service.sampler()

    results = sampler.run_sweep(program=circuit, params=param_resolvers, repetitions=repetitions)
    return results, quantum_computer, expected_results, param_resolvers


@pytest.mark.parametrize("result_builder", [_build_service_results, _build_sampler_results])
def test_parametric_circuit(
    mock_qpu_implementer: Any,
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable],
    result_builder: _ResultBuilder,
) -> None:
    """test that RigettiQCSService and RigettiQCSSampler can run a parametric
    circuit with a specified set of parameters and return expected cirq.Results.
    """

    parametric_circuit = parametric_circuit_with_params[0]
    sweepable = parametric_circuit_with_params[1]
    results, quantum_computer, expected_results, param_resolvers = result_builder(
        mock_qpu_implementer, parametric_circuit, sweepable
    )

    assert len(param_resolvers) == len(
        results
    ), "should return a result for every element in sweepable"

    for i, param_resolver in enumerate(param_resolvers):
        result = results[i]
        assert param_resolver == result.params
        assert np.allclose(
            result.measurements["m"], expected_results[i]
        ), "should return an ordered list of results with correct set of measurements"

    def test_executable(i: int, program: Program) -> None:
        params = param_resolvers[i]
        t = params["t"]
        if t == 1:
            assert (
                X(0) in program.instructions
            ), f"executable should contain an X(0) instruction at {i}"
        else:
            assert (
                RX(np.pi * t, 0) in program.instructions
            ), f"executable should contain an RX(pi*{t}) 0 instruction at {i}"
        assert DECLARE("m0") in program.instructions, "executable should declare a read out bit"
        assert (
            MEASURE(0, ("m0", 0)) in program.instructions
        ), "executable should measure the read out bit"

    param_sweeps = len(param_resolvers)
    assert param_sweeps == quantum_computer.compiler.quil_to_native_quil.call_count  # type: ignore
    for i, call_args in enumerate(
        quantum_computer.compiler.quil_to_native_quil.call_args_list  # type: ignore
    ):
        test_executable(i, call_args[0][0])

    assert (
        param_sweeps
        == quantum_computer.compiler.native_quil_to_executable.call_count  # type: ignore
    )
    for i, call_args in enumerate(
        quantum_computer.compiler.native_quil_to_executable.call_args_list  # type: ignore
    ):
        test_executable(i, call_args[0][0])

    assert param_sweeps == quantum_computer.qam.run.call_count  # type: ignore
    for i, call_args in enumerate(quantum_computer.qam.run.call_args_list):  # type: ignore
        test_executable(i, call_args[0][0])


@pytest.mark.parametrize("result_builder", [_build_service_results, _build_sampler_results])
def test_bell_circuit(
    mock_qpu_implementer: Any, bell_circuit: cirq.Circuit, result_builder: _ResultBuilder
) -> None:
    """test that RigettiQCSService and RigettiQCSSampler can run a basic Bell circuit
    with two read out bits and return expected cirq.Results.
    """

    param_resolvers = [cirq.ParamResolver({})]
    results, quantum_computer, expected_results, param_resolvers = result_builder(
        mock_qpu_implementer, bell_circuit, param_resolvers
    )

    assert len(param_resolvers) == len(
        results
    ), "should return a result for every element in sweepable"

    for i, param_resolver in enumerate(param_resolvers):
        result = results[i]
        assert param_resolver == result.params
        assert np.allclose(
            result.measurements["m"], expected_results[i]
        ), "should return an ordered list of results with correct set of measurements"

    def test_executable(program: Program) -> None:
        assert H(0) in program.instructions, "bell circuit should include Hadamard"
        assert CNOT(0, 1) in program.instructions, "bell circuit should include CNOT"
        assert (
            DECLARE("m0", memory_size=2) in program.instructions
        ), "executable should declare a read out bit"
        assert (
            MEASURE(0, ("m0", 0)) in program.instructions
        ), "executable should measure the first qubit to the first read out bit"
        assert (
            MEASURE(1, ("m0", 1)) in program.instructions
        ), "executable should measure the second qubit to the second read out bit"

    param_sweeps = len(param_resolvers)
    assert param_sweeps == quantum_computer.compiler.quil_to_native_quil.call_count  # type: ignore
    for i, call_args in enumerate(
        quantum_computer.compiler.quil_to_native_quil.call_args_list  # type: ignore
    ):
        test_executable(call_args[0][0])

    assert (
        param_sweeps
        == quantum_computer.compiler.native_quil_to_executable.call_count  # type: ignore
    )
    for i, call_args in enumerate(
        quantum_computer.compiler.native_quil_to_executable.call_args_list  # type: ignore
    ):
        test_executable(call_args[0][0])

    assert param_sweeps == quantum_computer.qam.run.call_count  # type: ignore
    for i, call_args in enumerate(quantum_computer.qam.run.call_args_list):  # type: ignore
        test_executable(call_args[0][0])


@pytest.mark.parametrize("result_builder", [_build_service_results, _build_sampler_results])
def test_explicit_qubit_id_map(
    mock_qpu_implementer: Any,
    bell_circuit_with_qids: Tuple[cirq.Circuit, List[cirq.LineQubit]],
    result_builder: _ResultBuilder,
) -> None:
    """test that RigettiQCSService and RigettiQCSSampler accept explicit ``qubit_id_map``
    to map ``cirq.Qid`` s to physical qubits.
    """
    bell_circuit, qubits = bell_circuit_with_qids

    qubit_id_map = {qubits[1]: "11", qubits[0]: "13"}
    param_resolvers = [cirq.ParamResolver({})]
    results, quantum_computer, expected_results, param_resolvers = result_builder(
        mock_qpu_implementer,
        bell_circuit,
        param_resolvers,
        transformer=transformers.build(qubit_id_map=qubit_id_map),  # type: ignore
    )

    assert len(param_resolvers) == len(
        results
    ), "should return a result for every element in sweepable"

    for i, param_resolver in enumerate(param_resolvers):
        result = results[i]
        assert param_resolver == result.params
        assert np.allclose(
            result.measurements["m"], expected_results[i]
        ), "should return an ordered list of results with correct set of measurements"

    def test_executable(program: Program) -> None:
        assert H(13) in program.instructions, "bell circuit should include Hadamard"
        assert CNOT(13, 11) in program.instructions, "bell circuit should include CNOT"
        assert (
            DECLARE("m0", memory_size=2) in program.instructions
        ), "executable should declare a read out bit"
        assert (
            MEASURE(13, ("m0", 0)) in program.instructions
        ), "executable should measure the first qubit to the first read out bit"
        assert (
            MEASURE(11, ("m0", 1)) in program.instructions
        ), "executable should measure the second qubit to the second read out bit"

    param_sweeps = len(param_resolvers)
    assert param_sweeps == quantum_computer.compiler.quil_to_native_quil.call_count  # type: ignore
    for i, call_args in enumerate(
        quantum_computer.compiler.quil_to_native_quil.call_args_list  # type: ignore
    ):
        test_executable(call_args[0][0])

    assert (
        param_sweeps
        == quantum_computer.compiler.native_quil_to_executable.call_count  # type: ignore
    )
    for i, call_args in enumerate(
        quantum_computer.compiler.native_quil_to_executable.call_args_list  # type: ignore
    ):
        test_executable(call_args[0][0])

    assert param_sweeps == quantum_computer.qam.run.call_count  # type: ignore
    for i, call_args in enumerate(quantum_computer.qam.run.call_args_list):  # type: ignore
        test_executable(call_args[0][0])


@pytest.mark.parametrize("result_builder", [_build_service_results, _build_sampler_results])
def test_run_without_quilc_compilation(
    mock_qpu_implementer: Any, bell_circuit: cirq.Circuit, result_builder: _ResultBuilder
) -> None:
    """test that RigettiQCSService and RigettiQCSSampler allow users to execute
    without using quilc to compile to native Quil.
    """
    param_resolvers = [cirq.ParamResolver({})]
    results, quantum_computer, expected_results, param_resolvers = result_builder(
        mock_qpu_implementer,
        bell_circuit,
        param_resolvers,
        executor=executors.without_quilc_compilation,
    )

    assert len(param_resolvers) == len(
        results
    ), "should return a result for every element in sweepable"

    for i, param_resolver in enumerate(param_resolvers):
        result = results[i]
        assert param_resolver == result.params
        assert np.allclose(
            result.measurements["m"], expected_results[i]
        ), "should return an ordered list of results with correct set of measurements"

    def test_executable(program: Program) -> None:
        assert H(0) in program.instructions, "bell circuit should include Hadamard"
        assert CNOT(0, 1) in program.instructions, "bell circuit should include CNOT"
        assert (
            DECLARE("m0", memory_size=2) in program.instructions
        ), "executable should declare a read out bit"
        assert (
            MEASURE(0, ("m0", 0)) in program.instructions
        ), "executable should measure the first qubit to the first read out bit"
        assert (
            MEASURE(1, ("m0", 1)) in program.instructions
        ), "executable should measure the second qubit to the second read out bit"

    assert 0 == quantum_computer.compiler.quil_to_native_quil.call_count  # type: ignore

    param_sweeps = len(param_resolvers)
    assert (
        param_sweeps
        == quantum_computer.compiler.native_quil_to_executable.call_count  # type: ignore
    )
    for i, call_args in enumerate(
        quantum_computer.compiler.native_quil_to_executable.call_args_list  # type: ignore
    ):
        test_executable(call_args[0][0])

    assert param_sweeps == quantum_computer.qam.run.call_count  # type: ignore
    for i, call_args in enumerate(quantum_computer.qam.run.call_args_list):  # type: ignore
        test_executable(call_args[0][0])

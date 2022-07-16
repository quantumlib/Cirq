# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import Tuple, Any, List, Union, Dict
import pytest
import cirq
from pyquil import Program
import numpy as np
import sympy
from cirq_rigetti import circuit_sweep_executors as executors, circuit_transformers


def test_with_quilc_compilation_and_cirq_parameter_resolution(
    mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]
) -> None:
    """test that execution with quilc compilation and cirq parameter resolution calls
    ``quil_to_native_quil`` and ``native_quil_to_executable`` for each parameter
    resolver.
    """

    parametric_circuit, sweepable = parametric_circuit_with_params
    repetitions = 2

    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [
        np.ones((repetitions,)) * (params["t"] if "t" in params else i)
        for i, params in enumerate(param_resolvers)
    ]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(
        expected_results
    )
    results = executors.with_quilc_compilation_and_cirq_parameter_resolution(
        quantum_computer=quantum_computer,
        circuit=parametric_circuit,
        resolvers=param_resolvers,  # ignore: type
        repetitions=repetitions,
    )
    assert len(param_resolvers) == len(results)
    assert len(param_resolvers) == quantum_computer.compiler.quil_to_native_quil.call_count
    assert len(param_resolvers) == quantum_computer.compiler.native_quil_to_executable.call_count

    for i, result in enumerate(results):
        result = results[i]
        assert param_resolvers[i] == result.params
        assert np.allclose(
            result.measurements["m"], expected_results[i]
        ), "should return an ordered list of results with correct set of measurements"


@pytest.mark.parametrize('pass_dict', [True, False])
def test_with_quilc_parametric_compilation(
    mock_qpu_implementer: Any,
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace],
    pass_dict: bool,
) -> None:
    """test that execution with quilc parametric compilation only compiles only once and
    parameters are properly resolved before execution.
    """

    parametric_circuit, sweepable = parametric_circuit_with_params
    repetitions = 2

    param_resolvers: List[Union[cirq.ParamResolver, cirq.ParamDictType]]
    if pass_dict:
        param_resolvers = [dict(params.param_dict) for params in sweepable]
    else:
        param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [
        np.ones((repetitions,)) * (params["t"] if "t" in params else i)
        for i, params in enumerate(param_resolvers)
    ]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(
        expected_results
    )
    results = executors.with_quilc_parametric_compilation(
        quantum_computer=quantum_computer,
        circuit=parametric_circuit,
        resolvers=param_resolvers,  # noqa
        repetitions=repetitions,
    )
    assert len(param_resolvers) == len(results)
    assert 1 == quantum_computer.compiler.quil_to_native_quil.call_count
    assert 1 == quantum_computer.compiler.native_quil_to_executable.call_count

    for i, result in enumerate(results):
        result = results[i]
        assert param_resolvers[i] == result.params
        assert np.allclose(
            result.measurements["m"], expected_results[i]
        ), "should return an ordered list of results with correct set of measurements"


def test_parametric_with_symbols(
    mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace]
):
    parametric_circuit, _ = parametric_circuit_with_params
    repetitions = 2
    expected_results = [np.ones((repetitions,))]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(
        expected_results
    )
    with pytest.raises(ValueError, match='Symbols not valid'):
        _ = executors.with_quilc_parametric_compilation(
            quantum_computer=quantum_computer,
            circuit=parametric_circuit,
            resolvers=[{sympy.Symbol('a') + sympy.Symbol('b'): sympy.Symbol('c')}],
            repetitions=repetitions,
        )


def test_without_quilc_compilation(
    mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]
) -> None:
    """test execution without quilc compilation treats the transformed cirq
    Circuit as native quil and does not pass it through quilc.
    """

    parametric_circuit, sweepable = parametric_circuit_with_params
    repetitions = 2

    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [
        np.ones((repetitions,)) * (params["t"] if "t" in params else i)
        for i, params in enumerate(param_resolvers)
    ]

    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(
        expected_results
    )
    results = executors.without_quilc_compilation(
        quantum_computer=quantum_computer,
        circuit=parametric_circuit,
        resolvers=param_resolvers,  # noqa
        repetitions=repetitions,
    )
    assert len(param_resolvers) == len(results)
    assert 0 == quantum_computer.compiler.quil_to_native_quil.call_count
    assert len(param_resolvers) == quantum_computer.compiler.native_quil_to_executable.call_count

    for i, result in enumerate(results):
        result = results[i]
        assert param_resolvers[i] == result.params
        assert np.allclose(
            result.measurements["m"], expected_results[i]
        ), "should return an ordered list of results with correct set of measurements"


def test_invalid_pyquil_region_measurement(
    mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]
) -> None:
    """test that executors raise `ValueError` if the measurement_id_map
    does not exist.
    """

    parametric_circuit, sweepable = parametric_circuit_with_params
    repetitions = 2

    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [
        np.ones((repetitions,)) * (params["t"] if "t" in params else i)
        for i, params in enumerate(param_resolvers)
    ]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(
        expected_results
    )

    def broken_hook(
        program: Program, measurement_id_map: Dict[str, str]
    ) -> Tuple[Program, Dict[str, str]]:
        return program, {cirq_key: f'{cirq_key}-doesnt-exist' for cirq_key in measurement_id_map}

    transformer = circuit_transformers.build(post_transformation_hooks=[broken_hook])

    with pytest.raises(ValueError):
        _ = executors.with_quilc_compilation_and_cirq_parameter_resolution(
            transformer=transformer,
            quantum_computer=quantum_computer,
            circuit=parametric_circuit,
            resolvers=param_resolvers,  # ignore: type
            repetitions=repetitions,
        )

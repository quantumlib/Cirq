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
"""A collection of `CircuitSweepExecutor` s that the client may pass to `RigettiQCSService` or
`RigettiQCSSampler` as `executor`.
"""

from typing import Any, cast, Dict, Optional, Sequence, Union

import sympy
from pyquil import Program
from pyquil.api import QuantumComputer, QuantumExecutable
from pyquil.quilbase import Declare
from typing_extensions import Protocol

import cirq
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti.deprecation import deprecated_cirq_rigetti_class, deprecated_cirq_rigetti_function
from cirq_rigetti.logging import logger


def _execute_and_read_result(
    quantum_computer: QuantumComputer,
    executable: QuantumExecutable,
    measurement_id_map: Dict[str, str],
    resolver: cirq.ParamResolverOrSimilarType,
    memory_map: Optional[
        Dict[Union[sympy.Expr, str], Union[int, float, Sequence[int], Sequence[float]]]
    ] = None,
) -> cirq.Result:
    """Execute the `pyquil.api.QuantumExecutable` and parse the measurements into
    a `cirq.Result`.

    Args:
        quantum_computer: The `pyquil.api.QuantumComputer` on which to execute
            and from which to read results.
        executable: The fully compiled `pyquil.api.QuantumExecutable` to run.
        measurement_id_map: A dict mapping cirq measurement keys to pyQuil
            read out regions.
        resolver: The `cirq.ParamResolverOrSimilarType` to include on
            the returned `cirq.Result`.
        memory_map: A dict of values to write to memory values on the
            `quantum_computer`. The `pyquil.api.QuantumAbstractMachine` reads these
            v_execute_and_read_resultalues into memory regions on the pre-compiled
            `executable` during execution.

    Returns:
        A `cirq.Result` with measurements read from the `quantum_computer`.

    Raises:
        ValueError: measurement_id_map references an undefined pyQuil readout region.
    """

    # convert all atomic memory values into 1-length lists
    if memory_map is not None:
        for region_name, value in memory_map.items():
            if not isinstance(region_name, str):
                raise ValueError(f'Symbols not valid for region name {region_name}')
            value = [value] if not isinstance(value, Sequence) else value
            memory_map[region_name] = value

    qam_execution_result = quantum_computer.qam.run(executable, memory_map)  # type: ignore

    measurements = {}
    # For every key, value in QuilOutput#measurement_id_map, use the value to read
    # Rigetti QCS results and assign to measurements by key.
    for cirq_memory_key, pyquil_region in measurement_id_map.items():
        readout = qam_execution_result.get_register_map().get(pyquil_region)
        if readout is None:
            raise ValueError(f'readout data does not have values for region "{pyquil_region}"')
        measurements[cirq_memory_key] = readout
    logger.debug(f"measurement_id_map {measurement_id_map}")
    logger.debug(f"measurements {measurements}")

    # collect results in a cirq.Result.
    result = cirq.ResultDict(
        params=cast(cirq.ParamResolver, resolver or cirq.ParamResolver({})),
        measurements=measurements,
    )  # noqa
    return result


def _get_param_dict(resolver: cirq.ParamResolverOrSimilarType) -> Dict[Union[str, sympy.Expr], Any]:
    """Converts a `cirq.ParamResolverOrSimilarType` to a dictionary.

    Args:
        resolver: A parameters resolver that provides values for parameter
            references in the original `cirq.Circuit`.
    Returns:
        A dictionary representation of the `resolver`.
    """
    param_dict: Dict[Union[str, sympy.Expr], Any] = {}
    if isinstance(resolver, cirq.ParamResolver):
        param_dict = dict(resolver.param_dict)
    elif isinstance(resolver, dict):
        param_dict = resolver
    return param_dict


def _prepend_real_declarations(
    *, program: Program, resolvers: Sequence[cirq.ParamResolverOrSimilarType]
) -> Program:
    """Adds memory declarations for all variables in each of the `resolver`'s
    param dict. Note, this function assumes that the first parameter resolver
    will contain all variables subsequently referenced in `resolvers`.

    Args:
        program: The program that the quantum computer will execute.
        resolvers: A sequence of parameters resolvers that provide values
            for parameter references in the original `cirq.Circuit`.
    Returns:
        A program that includes the QUIL memory declarations as described above.
    """
    if len(resolvers) > 0:
        resolver = resolvers[0]
        param_dict = _get_param_dict(resolver)
        for key in param_dict.keys():
            declaration = Declare(str(key), "REAL")
            program = Program(declaration) + program
            logger.debug(f"prepended declaration {declaration}")
    return program


@deprecated_cirq_rigetti_class()
class CircuitSweepExecutor(Protocol):
    """A type definition for circuit sweep execution functions."""

    def __call__(
        self,
        *,
        quantum_computer: QuantumComputer,
        circuit: cirq.Circuit,
        resolvers: Sequence[cirq.ParamResolverOrSimilarType],
        repetitions: int,
        transformer: transformers.CircuitTransformer,
    ) -> Sequence[cirq.Result]:
        """Transforms `cirq.Circuit` to `pyquil.Program` and executes it for given arguments.

        Args:
            quantum_computer: The `pyquil.api.QuantumComputer` against which to execute the circuit.
            circuit: The `cirq.Circuit` to transform into a `pyquil.Program` and executed on the
                `quantum_computer`.
            resolvers: A sequence of parameter resolvers that the executor must resolve.
            repetitions: Number of times to run each iteration through the `resolvers`. For a given
                resolver, the `cirq.Result` will include a measurement for each repetition.
            transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
                You may pass your own callable or any function from
                `cirq_rigetti.circuit_transformers`.

        Returns:
            A list of `cirq.Result`, each corresponding to a resolver in `resolvers`.
        """


@deprecated_cirq_rigetti_function()
def without_quilc_compilation(
    *,
    quantum_computer: QuantumComputer,
    circuit: cirq.Circuit,
    resolvers: Sequence[cirq.ParamResolverOrSimilarType],
    repetitions: int,
    transformer: transformers.CircuitTransformer = transformers.default,
) -> Sequence[cirq.Result]:
    """This `CircuitSweepExecutor` will bypass quilc entirely, treating the transformed
    `cirq.Circuit` as native Quil.

    Args:
        quantum_computer: The `pyquil.api.QuantumComputer` against which to execute the circuit.
        circuit: The `cirq.Circuit` to transform into a `pyquil.Program` and executed on the
            `quantum_computer`.
        resolvers: A sequence of parameter resolvers that `cirq.protocols.resolve_parameters` will
            use to fully resolve the circuit.
        repetitions: Number of times to run each iteration through the `resolvers`. For a given
            resolver, the `cirq.Result` will include a measurement for each repetition.
        transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
            You may pass your own callable or any function from `cirq_rigetti.circuit_transformers`.

    Returns:
        A list of `cirq.Result`, each corresponding to a resolver in `resolvers`.
    """

    cirq_results = []

    for resolver in resolvers:
        resolved_circuit = cirq.protocols.resolve_parameters(circuit, resolver)
        program, measurement_id_map = transformer(circuit=resolved_circuit)
        program = program.wrap_in_numshots_loop(repetitions)
        executable = quantum_computer.compile(program, optimize=False, to_native_gates=False)
        result = _execute_and_read_result(
            quantum_computer, executable, measurement_id_map, resolver
        )
        cirq_results.append(result)

    return cirq_results


@deprecated_cirq_rigetti_function()
def with_quilc_compilation_and_cirq_parameter_resolution(
    *,
    quantum_computer: QuantumComputer,
    circuit: cirq.Circuit,
    resolvers: Sequence[cirq.ParamResolverOrSimilarType],
    repetitions: int,
    transformer: transformers.CircuitTransformer = transformers.default,
) -> Sequence[cirq.Result]:
    """This `CircuitSweepExecutor` will first resolve each resolver in `resolvers` using
    `cirq.protocols.resolve_parameters` and then compile that resolved `cirq.Circuit` into
    native Quil using quilc. This executor may be useful if `with_quilc_parametric_compilation`
    fails to properly resolve a parameterized `cirq.Circuit`.

    Args:
        quantum_computer: The `pyquil.api.QuantumComputer` against which to execute the circuit.
        circuit: The `cirq.Circuit` to transform into a `pyquil.Program` and executed on the
            `quantum_computer`.
        resolvers: A sequence of parameter resolvers that `cirq.protocols.resolve_parameters` will
            use to fully resolve the circuit.
        repetitions: Number of times to run each iteration through the `resolvers`. For a given
            resolver, the `cirq.Result` will include a measurement for each repetition.
        transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
            You may pass your own callable or any function from `cirq_rigetti.circuit_transformers`.

    Returns:
        A list of `cirq.Result`, each corresponding to a resolver in `resolvers`.
    """

    cirq_results = []
    for resolver in resolvers:
        resolved_circuit = cirq.protocols.resolve_parameters(circuit, resolver)
        program, measurement_id_map = transformer(circuit=resolved_circuit)
        program.wrap_in_numshots_loop(repetitions)

        executable = quantum_computer.compile(program)
        result = _execute_and_read_result(
            quantum_computer, executable, measurement_id_map, resolver
        )
        cirq_results.append(result)

    return cirq_results


@deprecated_cirq_rigetti_function()
def with_quilc_parametric_compilation(
    *,
    quantum_computer: QuantumComputer,
    circuit: cirq.Circuit,
    resolvers: Sequence[cirq.ParamResolverOrSimilarType],
    repetitions: int,
    transformer: transformers.CircuitTransformer = transformers.default,
) -> Sequence[cirq.Result]:
    """This `CircuitSweepExecutor` will compile the `circuit` using quilc as a
    parameterized `pyquil.api.QuantumExecutable` and on each iteration of
    `resolvers`, rather than resolving the `circuit` with `cirq.protocols.resolve_parameters`,
    it will attempt to cast the resolver to a dict and pass it as a memory map to
    to `pyquil.api.QuantumComputer`.

    Args:
        quantum_computer: The `pyquil.api.QuantumComputer` against which to execute the circuit.
        circuit: The `cirq.Circuit` to transform into a `pyquil.Program` and executed on the
            `quantum_computer`.
        resolvers: A sequence of parameter resolvers that this executor will write to memory
            on a copy of the `pyquil.api.QuantumExecutable` for each parameter sweep.
        repetitions: Number of times to run each iteration through the `resolvers`. For a given
            resolver, the `cirq.Result` will include a measurement for each repetition.
        transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
            You may pass your own callable or any function from `cirq_rigetti.circuit_transformers`.

    Returns:
        A list of `cirq.Result`, each corresponding to a resolver in `resolvers`.
    """

    program, measurement_id_map = transformer(circuit=circuit)
    program = _prepend_real_declarations(program=program, resolvers=resolvers)
    program.wrap_in_numshots_loop(repetitions)
    executable = quantum_computer.compile(program)

    cirq_results = []
    for resolver in resolvers:
        memory_map = _get_param_dict(resolver)
        logger.debug(f"running pre-compiled parametric circuit with parameters {memory_map}")
        result = _execute_and_read_result(
            quantum_computer, executable.copy(), measurement_id_map, resolver, memory_map=memory_map
        )
        cirq_results.append(result)

    return cirq_results

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
"""A collection of `CircuitTransformer` s that the client may pass to `RigettiQCSService` or
`RigettiQCSSampler` as `transformer`.
"""
from typing import Dict, cast, Optional, Tuple, List, Callable
from pyquil import Program
import cirq
from cirq_rigetti._quil_output import RigettiQCSQuilOutput
from typing_extensions import Protocol


class _PostTransformationHook(Protocol):
    def __call__(
        self, *, program: Program, measurement_id_map: Dict[str, str]
    ) -> Tuple[Program, Dict[str, str]]:
        pass


def _transform_cirq_circuit_to_pyquil_program(
    *,
    circuit: cirq.Circuit,
    qubits: Optional[Tuple[cirq.Qid, ...]] = None,
    decompose_operation: Optional[Callable[[cirq.Operation], List[cirq.Operation]]] = None,
    qubit_id_map: Optional[Dict[cirq.Qid, str]] = None,
    post_transformation_hooks: Optional[List[_PostTransformationHook]] = None,
) -> Tuple[Program, Dict[str, str]]:
    if qubits is None:
        qubits = cirq.QubitOrder.as_qubit_order(cirq.ops.QubitOrder.DEFAULT).order_for(
            circuit.all_qubits()
        )
    quil_output = RigettiQCSQuilOutput(
        operations=circuit.all_operations(),
        qubits=qubits,
        qubit_id_map=qubit_id_map,
        decompose_operation=decompose_operation,
    )

    program = Program(str(quil_output))
    measurement_id_map = quil_output.measurement_id_map
    if post_transformation_hooks is not None:
        for hook in post_transformation_hooks:
            program, measurement_id_map = hook(
                program=program, measurement_id_map=measurement_id_map
            )
    return program, measurement_id_map


class CircuitTransformer(Protocol):
    """A type definition for `cirq.Circuit` to `pyquil.Program` transformer functions."""

    def __call__(self, *, circuit: cirq.Circuit) -> Tuple[Program, Dict[str, str]]:
        """Transforms a `cirq.Circuit` to a pyquil.Program`.

        Args:
            circuit: The `cirq.Circuit` the transformer will transform into a `pyquil.Program`.

        Returns:
            The `pyquil.Program` and a map of the `cirq.Circuit`'s memory region keys to
            the `pyquil.Program`'s memory regions.
        """


def build(
    *,
    qubits: Optional[Tuple[cirq.Qid, ...]] = None,
    decompose_operation: Optional[Callable[[cirq.Operation], List[cirq.Operation]]] = None,
    qubit_id_map: Optional[Dict[cirq.Qid, str]] = None,
    post_transformation_hooks: Optional[List[_PostTransformationHook]] = None,
) -> CircuitTransformer:
    """This builds a `CircuitTransformer` that the client may use over multiple sweeps of
    `cirq.Sweepable`.

    Args:
        qubits: The qubits defined on the circuit that this function will transform. If None,
            the transformer will pull qubits from the `cirq.Circuit` and order them by
            `cirq.ops.QubitOrder.DEFAULT` on each transformation.
        decompose_operation: A callable that can decompose each individual operation on the
            `cirq.Circuit` before being transformed. This will override the default Quil
            decompositions in cirq. You may optimize your circuit before transformation and pass
            a no-op here.
        qubit_id_map: A map of `cirq.Qid` to physical qubit addresses that will end up in
            the executed native Quil.
        post_transformation_hooks: A list of transformation functions you may pass to further
            convert a `pyquil.Program` after transformation.

    Returns:
        A `CircuitTransformer` transforming the `cirq.Circuit` s as specified above.
    """

    def transformer(*, circuit: cirq.Circuit) -> Tuple[Program, Dict[str, str]]:
        return _transform_cirq_circuit_to_pyquil_program(
            circuit=circuit,
            qubits=qubits,
            decompose_operation=decompose_operation,
            qubit_id_map=qubit_id_map,
            post_transformation_hooks=post_transformation_hooks,
        )

    return cast(CircuitTransformer, transformer)


def default(*, circuit: cirq.Circuit) -> Tuple[Program, Dict[str, str]]:
    """The default `CircuitTransformer` uses the default behavior of cirq's Quil
    protocol to transform a `cirq.Circuit` into a `pyquil.Program`.

    Args:
        circuit: The `cirq.Circuit` the transformer will transform into a `pyquil.Program`.

    Returns:
        The `pyquil.Program` and a map of the `cirq.Circuit`'s memory region keys to
        the `pyquil.Program`'s memory regions.
    """
    return _transform_cirq_circuit_to_pyquil_program(circuit=circuit)

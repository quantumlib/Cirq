# Copyright 2022 The Cirq Developers
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

"""Target gateset used for compiling circuits to Sycamore + 1-q rotations + measurement gates."""

import itertools
from typing import cast, Any, Dict, List, Optional, Sequence

import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq_google import ops
from cirq_google.transformers.analytical_decompositions import two_qubit_to_sycamore


@cirq.transformer
def merge_swap_rzz_and_2q_unitaries(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    merged_swap_rzz_tag: str = "_merged_swap_rzz",
    merged_2q_component_tag: str = "_merged_2q_unitaries",
    intermediate_result_tag: Optional[str] = None,
) -> 'cirq.Circuit':
    """Merges 2-qubit connected components and adjacent `cirq.SWAP` and `cirq.ZZPowGate` gates.

    Does the following two transformations, in that order:
    1. Merges adjacent occurrences of `cirq.SWAP` and `cirq.ZZPowGate` into a
    `cirq.CircuitOperation` tagged with `merged_swap_rzz_tag`.
    2. Merges connected components of 1 and 2 qubit unitaries in the circuit into a
    `cirq.CircuitOperation` tagged with `merged_2q_component_tag`, ignoring the newly
    introduced tagged circuit operations added in Step-1.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        merged_swap_rzz_tag: Tag to apply on newly introduced circuit operations wrapping adjacent
            `cirq.SWAP` and `cirq.ZZPowGate`s.
        merged_2q_component_tag: Tag to apply on newly introduced circuit operations wrapping
            connected components of 1 and 2 qubit unitaries.
        intermediate_result_tag: If specified, the tag is added to newly introduced both the newly
            introduced circuit operations encapsulating swap_rzz or 2q connected component.

    Returns:
        Copy of the transformed input circuit.

    Raises:
          ValueError: If merged_2q_component_tag and merged_swap_rzz_tag are equal.
    """
    if merged_2q_component_tag == merged_swap_rzz_tag:
        raise ValueError("merged_swap_rzz_tag and merged_2q_component_tag should be different.")

    def merge_func_swap_rzz(
        ops1: Sequence['cirq.Operation'], ops2: Sequence['cirq.Operation']
    ) -> bool:
        if not (len(ops1) == 1 and len(ops2) == 1):
            return False
        for op1, op2 in itertools.permutations([ops1[0], ops2[0]]):
            if op1.gate == cirq.SWAP and isinstance(op2.gate, cirq.ZZPowGate):
                return True
        return False

    tags_to_ignore = context.tags_to_ignore if context else ()
    deep = context.deep if context else False
    circuit = cirq.merge_operations_to_circuit_op(
        circuit,
        merge_func_swap_rzz,
        tags_to_ignore=tags_to_ignore,
        merged_circuit_op_tag=merged_swap_rzz_tag,
        deep=deep,
    )

    circuit = cirq.merge_k_qubit_unitaries_to_circuit_op(
        circuit,
        k=2,
        tags_to_ignore=tuple(tags_to_ignore) + (merged_swap_rzz_tag,),
        merged_circuit_op_tag=merged_2q_component_tag,
        deep=deep,
    )

    if intermediate_result_tag is not None:
        merged_cop_tags = {merged_swap_rzz_tag, merged_2q_component_tag}
        circuit = cirq.map_operations(
            circuit,
            map_func=lambda op, _: (
                op
                if merged_cop_tags.isdisjoint(op.tags)
                else op.with_tags(cast(str, intermediate_result_tag))
            ),
            tags_to_ignore=tags_to_ignore,
            deep=True,
        )
    return circuit.unfreeze(copy=False)


class SycamoreTargetGateset(cirq.TwoQubitCompilationTargetGateset):
    """Target gateset containing Sycamore + single qubit rotations + Measurement gates."""

    def __init__(
        self, *, atol: float = 1e-8, tabulation: Optional[cirq.TwoQubitGateTabulation] = None
    ) -> None:
        """Inits `cirq_google.SycamoreTargetGateset`.

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            tabulation: If set, a tabulation for the Sycamore gate is used for decomposing Matrix
                gates. If unset, an analytic calculation is used for Matrix gates. In both cases,
                known decompositions for gates take priority over analytical / tabulation methods.
                To get `cirq.TwoQubitGateTabulation`, call `cirq.two_qubit_gate_product_tabulation`
                with a base gate (in this case, `cirq_google.SYC`) and a maximum infidelity.
        """
        super().__init__(
            ops.SYC,
            cirq.MeasurementGate,
            cirq.PhasedXZGate,
            cirq.PhasedXPowGate,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.GlobalPhaseGate,
            name='SycamoreTargetGateset',
        )
        self.atol = atol
        self.tabulation = tabulation

    @property
    def preprocess_transformers(self) -> List[cirq.TRANSFORMER]:
        return [
            cirq.create_transformer_with_kwargs(
                cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= self.num_qubits
            ),
            cirq.create_transformer_with_kwargs(
                merge_swap_rzz_and_2q_unitaries,
                intermediate_result_tag=self._intermediate_result_tag,
            ),
        ]

    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> DecomposeResult:
        if not cirq.has_unitary(op):
            return NotImplemented

        known_decomp = two_qubit_to_sycamore.known_2q_op_to_sycamore_operations(op)
        if known_decomp is not None:
            return known_decomp
        if self.tabulation is not None:
            return two_qubit_to_sycamore._decompose_arbitrary_into_syc_tabulation(
                op, self.tabulation
            )
        return two_qubit_to_sycamore.two_qubit_matrix_to_sycamore_operations(
            op.qubits[0], op.qubits[1], cirq.unitary(op)
        )

    def __repr__(self) -> str:
        return (
            f'cirq_google.SycamoreTargetGateset('
            f'atol={self.atol}, '
            f'tabulation={self.tabulation})'
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {'atol': self.atol, 'tabulation': self.tabulation}

    @classmethod
    def _from_json_dict_(cls, atol, tabulation, **kwargs):
        return cls(atol=atol, tabulation=tabulation)

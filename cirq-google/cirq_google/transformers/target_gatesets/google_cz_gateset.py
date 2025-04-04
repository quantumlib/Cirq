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

from typing import Any, Dict, List, Sequence, Type, Union

import cirq


class GoogleCZTargetGateset(cirq.CZTargetGateset):
    """`cirq.CZTargetGateset` implementation tailored to Google devices.

    In addition to features available from `cirq.CZTargetGateset`, `GoogleCZTargetGateset` contains
    a flag, `eject_paulis`, to enable the postprocess transformers `cirq.eject_phased_paulis` and
    `cirq.eject_z`, which will push X, Y, Z, PhasedX, and certain PhasedXZ gates to the end of the
    circuit.
    """

    def __init__(
        self,
        atol: float = 1e-8,
        eject_paulis: bool = False,
        additional_gates: Sequence[Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily]] = (),
    ):
        """Initializes GoogleCZTargetGateset.

        Args:
            atol: A limit on the amount of absolute error introduced by the transformation.
            eject_paulis: Whether to enable postprocess transformers `cirq.eject_z` and
             `cirq.eject_phased_paulis`. If enabled, these transformers will remove tags (e.g.
             `cirq_google.PhysicalZTag`) from single-qubit Pauli operations. Defaults to False.
            additional_gates: Sequence of additional gates / gate families which should also
              be "accepted" by this gateset. This is empty by default.
        """
        super().__init__(atol=atol, allow_partial_czs=False, additional_gates=additional_gates)
        self.eject_paulis = eject_paulis
        self._additional_gates_repr_str = ", ".join(
            [cirq.ops.gateset._gate_str(g, repr) for g in additional_gates]
        )

    @property
    def postprocess_transformers(self) -> List[cirq.TRANSFORMER]:
        """List of transformers which should be run after decomposing individual operations.

        If `eject_paulis` is enabled in the constructor, adds `cirq.eject_phased_paulis` and
        `cirq.eject_z` in addition to postprocess_transformers already available in
        `cirq.CompilationTargetGateset`.
        """
        transformers: List[cirq.TRANSFORMER] = [
            cirq.create_transformer_with_kwargs(
                cirq.merge_single_qubit_moments_to_phxz, atol=self.atol
            ),
            cirq.create_transformer_with_kwargs(cirq.drop_negligible_operations, atol=self.atol),
            cirq.drop_empty_moments,
        ]

        if self.eject_paulis:
            return (
                transformers[:1]
                + [
                    cirq.create_transformer_with_kwargs(cirq.eject_phased_paulis, atol=self.atol),
                    cirq.create_transformer_with_kwargs(cirq.eject_z, atol=self.atol),
                ]
                + transformers[1:]
            )
        return transformers

    def __repr__(self) -> str:
        return (
            'cirq_google.GoogleCZTargetGateset('
            f'atol={self.atol}, '
            f'eject_paulis={self.eject_paulis}, '
            f'additional_gates=[{self._additional_gates_repr_str}]'
            ')'
        )

    def _value_equality_values_(self) -> Any:
        return self.atol, self.eject_paulis, frozenset(self.additional_gates)

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'atol': self.atol,
            'eject_paulis': self.eject_paulis,
            'additional_gates': list(self.additional_gates),
        }

    @classmethod
    def _from_json_dict_(cls, atol, eject_paulis, additional_gates, **kwargs):
        return cls(atol=atol, eject_paulis=eject_paulis, additional_gates=additional_gates)

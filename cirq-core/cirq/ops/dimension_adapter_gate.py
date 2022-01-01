# Copyright 2018 The Cirq Developers
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

from typing import AbstractSet, Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

import cirq
from cirq import protocols, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class DimensionAdapterGate(raw_types.Gate):
    """Adapts a 2-qubit gate to apply to qudits."""

    def __init__(
        self, gate: 'cirq.Gate', subspaces: Sequence[Tuple[int, Union[Tuple[int, int], slice]]]
    ):
        """Initializes the adapter gate.

        Args:
            gate: The gate to add a control qubit to.
            subspaces: : The subspaces of the qudits to use. The sequence is a
                tuple of (dimension_of_qudit, slice_of_that_dimension_to_use).
                So to adapt an X gate to the (0, 2) subspace of a qutrit, one
                would write `DimensionAdapterGate(X, [(3, slice(0, 3, 2)])`.

        Raises:
            ValueError: If the gate's qubit count does not match the subspace
                count, if the dimensions of the subspaces do not match the
                dimensions of the gate on any qubit, or if the subspaces are
                outside the range of the input qudit dimensions.
        """
        if gate.num_qubits() != len(subspaces):
            raise ValueError('gate qubit count and subspace count must match')

        def to_slice(subspace: Union[Tuple[int, int], slice]):
            if isinstance(subspace, slice):
                return subspace
            subspace0, subspace1 = subspace
            step = subspace1 - subspace0
            stop = subspace1 + step
            return slice(subspace0, stop if stop >= 0 else None, step)

        gate_shape = protocols.qid_shape(gate)
        slices = []
        shape = []
        for i in range(len(subspaces)):
            qubit_dimension, slice_def = subspaces[i]
            sleis = to_slice(slice_def)
            subspace = list(range(qubit_dimension))[sleis]
            if max(subspace) >= qubit_dimension:
                raise ValueError(
                    f'dimension {qubit_dimension}'
                    f' not large enough for subspace {subspace} on qubit {i}'
                )
            gate_dimension = gate_shape[i]
            if len(subspace) != gate_dimension:
                raise ValueError(
                    f'subspace {subspace}'
                    f' does not match gate dimension {gate_dimension} on qubit {i}'
                )
            shape.append(qubit_dimension)
            slices.append(sleis)

        self._gate = gate
        self._slices = tuple(slices)
        self._shape = tuple(shape)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._shape

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        my_args = protocols.ApplyUnitaryArgs(
            target_tensor=args.target_tensor,
            available_buffer=args.available_buffer,
            slices=self._slices,
            axes=args.axes,
        )
        return protocols.apply_unitary(self._gate, args=my_args)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        # TODO
        return NotImplemented

    def _value_equality_values_(self):
        return (self._gate, self._slices, self._shape)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self._gate)

    def __pow__(self, exponent: Any) -> 'DimensionAdapterGate':
        new_gate = protocols.pow(self._gate, exponent, NotImplemented)
        if new_gate is NotImplemented:
            return NotImplemented
        return DimensionAdapterGate(new_gate, tuple(zip(self._shape, self._slices)))

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._gate)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'DimensionAdapterGate':
        new_gate = protocols.resolve_parameters(self._gate, resolver, recursive)
        return DimensionAdapterGate(new_gate, tuple(zip(self._shape, self._slices)))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        sub_info = protocols.circuit_diagram_info(self._gate, args, None)
        if sub_info is None:
            return NotImplemented

        # TODO
        return sub_info

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return (
            f'cirq.DimensionAdapterGate(gate={self._gate!r}, '
            f'subspaces={list(zip(self._shape, self._slices))!r})'
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'subspaces': list(zip(self._shape, self._slices)),
            'gate': self._gate,
        }

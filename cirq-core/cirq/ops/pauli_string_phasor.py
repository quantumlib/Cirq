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

from typing import AbstractSet, Dict, Iterable, Iterator, Sequence, Union, TYPE_CHECKING

import sympy

from cirq import value, protocols
from cirq._compat import proper_repr, deprecated
from cirq.ops import (
    raw_types,
    clifford_gate,
    common_gates,
    dense_pauli_string as dps,
    pauli_string as ps,
    pauli_gates,
    op_tree,
    pauli_string_raw_types,
)

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliStringPhasor(pauli_string_raw_types.PauliStringGateOperation):
    """An operation that phases the eigenstates of a Pauli string.

    The -1 eigenstates of the Pauli string will have their amplitude multiplied
    by e^(i pi exponent_neg) while +1 eigenstates of the Pauli string will have
    their amplitude multiplied by e^(i pi exponent_pos).
    """

    def __init__(
        self,
        pauli_string: ps.PauliString,
        *,
        exponent_neg: Union[int, float, sympy.Basic] = 1,
        exponent_pos: Union[int, float, sympy.Basic] = 0,
    ) -> None:
        """Initializes the operation.

        Args:
            pauli_string: The PauliString defining the positive and negative
                eigenspaces that will be independently phased.
            exponent_neg: How much to phase vectors in the negative eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).
            exponent_pos: How much to phase vectors in the positive eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).

        Raises:
            ValueError: If the given pauli string does not have eignevalues +1
                or -1.
        """
        if pauli_string.coefficient == -1:
            pauli_string = -pauli_string
            exponent_pos, exponent_neg = exponent_neg, exponent_pos

        if pauli_string.coefficient != 1:
            raise ValueError(
                "Given PauliString doesn't have +1 and -1 eigenvalues. "
                "pauli_string.coefficient must be 1 or -1."
            )

        super().__init__(pauli_string)
        self._gate = PauliStringPhasorGate(
            dps.DensePauliString(pauli_string.values(), coefficient=pauli_string.coefficient),
            exponent_neg=exponent_neg,
            exponent_pos=exponent_pos,
        )

    @property
    def gate(self):
        return self._gate

    @property
    def exponent_neg(self):
        return self._gate.exponent_neg

    @exponent_neg.setter  # type: ignore
    @deprecated(
        deadline="v0.15",
        fix="The mutators of this class are deprecated, instantiate a new object instead.",
    )
    def exponent_neg(self, exponent_neg):
        # coverage: ignore
        self._gate._exponent_neg = exponent_neg

    @property
    def exponent_pos(self):
        return self._gate.exponent_pos

    @exponent_pos.setter  # type: ignore
    @deprecated(
        deadline="v0.15",
        fix="The mutators of this class are deprecated, instantiate a new object instead.",
    )
    def exponent_pos(self, exponent_pos):
        # coverage: ignore
        self._gate._exponent_pos = exponent_pos

    @deprecated(
        deadline="v0.15",
        fix="This is a temporary shim until the mutator is deprecated.",
    )
    def _on_pauli_string_changed(self, pauli_string: 'cirq.PauliString'):
        # coverage: ignore
        self._gate._dense_pauli_string = dps.DensePauliString(  # type: ignore
            pauli_string.values(), coefficient=pauli_string.coefficient
        )

    @property
    def exponent_relative(self) -> Union[int, float, sympy.Basic]:
        return self._gate.exponent_relative

    def _value_equality_values_(self):
        return (
            self.pauli_string,
            self.exponent_neg,
            self.exponent_pos,
        )

    def equal_up_to_global_phase(self, other):
        if isinstance(other, PauliStringPhasor):
            return self._gate.equal_up_to_global_phase(other.gate)
        return False

    def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]):
        return PauliStringPhasor(
            self.pauli_string.map_qubits(qubit_map),
            exponent_neg=self.exponent_neg,
            exponent_pos=self.exponent_pos,
        )

    def __pow__(self, exponent: Union[float, sympy.Symbol]) -> 'PauliStringPhasor':
        pn = protocols.mul(self.exponent_neg, exponent, None)
        pp = protocols.mul(self.exponent_pos, exponent, None)
        if pn is None or pp is None:
            return NotImplemented
        return PauliStringPhasor(self.pauli_string, exponent_neg=pn, exponent_pos=pp)

    def can_merge_with(self, op: 'PauliStringPhasor') -> bool:
        return self.pauli_string.equal_up_to_coefficient(op.pauli_string)

    def merged_with(self, op: 'PauliStringPhasor') -> 'PauliStringPhasor':
        if not self.can_merge_with(op):
            raise ValueError(f'Cannot merge operations: {self}, {op}')
        pp = self.exponent_pos + op.exponent_pos
        pn = self.exponent_neg + op.exponent_neg
        return PauliStringPhasor(self.pauli_string, exponent_pos=pp, exponent_neg=pn)

    def _has_unitary_(self):
        return self._gate._has_unitary_()

    def _decompose_(self) -> 'cirq.OP_TREE':
        return protocols.decompose_once_with_qubits(self.gate, self.qubits, NotImplemented)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return self._pauli_string_diagram_info(args, exponent=self.exponent_relative)

    def _trace_distance_bound_(self) -> float:
        return self._gate._trace_distance_bound_()

    def _is_parameterized_(self) -> bool:
        return self._gate._is_parameterized_()

    def _parameter_names_(self) -> AbstractSet[str]:
        return self._gate._parameter_names_()

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'PauliStringPhasor':
        return PauliStringPhasor(
            self.pauli_string,
            exponent_neg=resolver.value_of(self.exponent_neg, recursive),
            exponent_pos=resolver.value_of(self.exponent_pos, recursive),
        )

    def pass_operations_over(
        self, ops: Iterable[raw_types.Operation], after_to_before: bool = False
    ) -> 'PauliStringPhasor':
        new_pauli_string = self.pauli_string.pass_operations_over(ops, after_to_before)
        pp = self.exponent_pos
        pn = self.exponent_neg
        return PauliStringPhasor(new_pauli_string, exponent_pos=pp, exponent_neg=pn)

    def __repr__(self) -> str:
        return (
            f'cirq.PauliStringPhasor({self.pauli_string!r}, '
            f'exponent_neg={proper_repr(self.exponent_neg)}, '
            f'exponent_pos={proper_repr(self.exponent_pos)})'
        )

    def __str__(self) -> str:
        if self.exponent_pos == -self.exponent_neg:
            sign = '-' if self.exponent_pos < 0 else ''
            exponent = str(abs(self.exponent_pos))
            return f'exp({sign}iπ{exponent}*{self.pauli_string})'
        return f'({self.pauli_string})**{self.exponent_relative}'


@value.value_equality(approximate=True)
class PauliStringPhasorGate(raw_types.Gate):
    """A gate that phases the eigenstates of a Pauli string.

    The -1 eigenstates of the Pauli string will have their amplitude multiplied
    by e^(i pi exponent_neg) while +1 eigenstates of the Pauli string will have
    their amplitude multiplied by e^(i pi exponent_pos).
    """

    def __init__(
        self,
        dense_pauli_string: dps.DensePauliString,
        *,
        exponent_neg: Union[int, float, sympy.Basic] = 1,
        exponent_pos: Union[int, float, sympy.Basic] = 0,
    ) -> None:
        """Initializes the PauliStringPhasorGate.

        Args:
            dense_pauli_string: The DensePauliString defining the positive and
                negative eigenspaces that will be independently phased.
            exponent_neg: How much to phase vectors in the negative eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).
            exponent_pos: How much to phase vectors in the positive eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).

        Raises:
            ValueError: If coefficient is not 1 or -1.
        """
        if dense_pauli_string.coefficient == -1:
            dense_pauli_string = -dense_pauli_string
            exponent_pos, exponent_neg = exponent_neg, exponent_pos

        if dense_pauli_string.coefficient != 1:
            raise ValueError(
                "Given PauliString doesn't have +1 and -1 eigenvalues. "
                "pauli_string.coefficient must be 1 or -1."
            )

        self._dense_pauli_string = dense_pauli_string
        self._exponent_neg = value.canonicalize_half_turns(exponent_neg)
        self._exponent_pos = value.canonicalize_half_turns(exponent_pos)

    @property
    def exponent_relative(self) -> Union[int, float, sympy.Basic]:
        return value.canonicalize_half_turns(self._exponent_neg - self._exponent_pos)

    @property
    def exponent_neg(self):
        return self._exponent_neg

    @property
    def exponent_pos(self):
        return self._exponent_pos

    @property
    def dense_pauli_string(self):
        return self._dense_pauli_string

    def _value_equality_values_(self):
        return (
            self._dense_pauli_string,
            self._exponent_neg,
            self._exponent_pos,
        )

    def equal_up_to_global_phase(self, other):
        if isinstance(other, PauliStringPhasorGate):
            rel1 = self.exponent_relative
            rel2 = other.exponent_relative
            return rel1 == rel2 and self._dense_pauli_string == other.dense_pauli_string
        return False

    def __pow__(self, exponent: Union[float, sympy.Symbol]) -> 'PauliStringPhasorGate':
        pn = protocols.mul(self._exponent_neg, exponent, None)
        pp = protocols.mul(self._exponent_pos, exponent, None)
        if pn is None or pp is None:
            return NotImplemented
        return PauliStringPhasorGate(self._dense_pauli_string, exponent_neg=pn, exponent_pos=pp)

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _to_z_basis_ops(self, qubits: Sequence['cirq.Qid']) -> Iterator[raw_types.Operation]:
        """Returns operations to convert the qubits to the computational basis."""
        for i in range(len(self._dense_pauli_string)):
            yield clifford_gate.SingleQubitCliffordGate.from_single_map(
                {self._dense_pauli_string[i]: (pauli_gates.Z, False)}
            )(qubits[i])

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        if len(self._dense_pauli_string) <= 0:
            return
        any_qubit = qubits[0]
        to_z_ops = op_tree.freeze_op_tree(self._to_z_basis_ops(qubits))
        xor_decomp = tuple(xor_nonlocal_decompose(qubits, any_qubit))
        yield to_z_ops
        yield xor_decomp

        if self._exponent_neg:
            yield pauli_gates.Z(any_qubit) ** self._exponent_neg
        if self._exponent_pos:
            yield pauli_gates.X(any_qubit)
            yield pauli_gates.Z(any_qubit) ** self._exponent_pos
            yield pauli_gates.X(any_qubit)

        yield protocols.inverse(xor_decomp)
        yield protocols.inverse(to_z_ops)

    def _trace_distance_bound_(self) -> float:
        if len(self._dense_pauli_string) == 0:
            return 0.0
        return protocols.trace_distance_bound(pauli_gates.Z ** self.exponent_relative)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._exponent_neg) or protocols.is_parameterized(
            self._exponent_pos
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._exponent_neg) | protocols.parameter_names(
            self._exponent_pos
        )

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'PauliStringPhasorGate':
        return PauliStringPhasorGate(
            self._dense_pauli_string,
            exponent_neg=resolver.value_of(self._exponent_neg, recursive),
            exponent_pos=resolver.value_of(self._exponent_pos, recursive),
        )

    def __repr__(self) -> str:
        return (
            f'cirq.PauliStringPhasorGate({self._dense_pauli_string!r}, '
            f'exponent_neg={proper_repr(self.exponent_neg)}, '
            f'exponent_pos={proper_repr(self.exponent_pos)})'
        )

    def __str__(self) -> str:
        if self._exponent_pos == -self._exponent_neg:
            sign = '-' if self._exponent_pos < 0 else ''
            exponent = str(abs(self._exponent_pos))
            return f'exp({sign}iπ{exponent}*{self._dense_pauli_string})'
        return f'({self._dense_pauli_string})**{self.exponent_relative}'

    def num_qubits(self) -> int:
        return len(self._dense_pauli_string)

    def on(self, *qubits: 'cirq.Qid') -> 'cirq.PauliStringPhasor':
        return PauliStringPhasor(
            self._dense_pauli_string.on(*qubits),
            exponent_pos=self._exponent_pos,
            exponent_neg=self._exponent_neg,
        )


def xor_nonlocal_decompose(
    qubits: Iterable[raw_types.Qid], onto_qubit: 'cirq.Qid'
) -> Iterable[raw_types.Operation]:
    """Decomposition ignores connectivity."""
    for qubit in qubits:
        if qubit != onto_qubit:
            yield common_gates.CNOT(qubit, onto_qubit)

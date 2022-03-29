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

from typing import AbstractSet, cast, Dict, Iterable, Union, TYPE_CHECKING, Sequence, Iterator

import sympy

from cirq import value, protocols
from cirq._compat import proper_repr, deprecated
from cirq.ops import (
    raw_types,
    common_gates,
    gate_operation,
    dense_pauli_string as dps,
    pauli_string as ps,
    pauli_gates,
    op_tree,
)

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliStringPhasor(gate_operation.GateOperation):
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
            ValueError: If coefficient is not 1 or -1.
        """
        gate = PauliStringPhasorGate(
            pauli_string.dense(pauli_string.qubits),
            exponent_neg=exponent_neg,
            exponent_pos=exponent_pos,
        )
        super().__init__(gate, pauli_string.qubits)
        self._pauli_string = gate.dense_pauli_string.on(*self.qubits)

    @property
    def gate(self) -> 'cirq.PauliStringPhasorGate':
        """The gate applied by the operation."""
        return cast(PauliStringPhasorGate, self._gate)

    @property
    def exponent_neg(self):
        """The negative exponent."""
        return self.gate.exponent_neg

    @exponent_neg.setter  # type: ignore
    @deprecated(
        deadline="v0.15",
        fix="The mutators of this class are deprecated, instantiate a new object instead.",
    )
    def exponent_neg(self, exponent_neg):
        """Sets the negative exponent."""
        # coverage: ignore
        self.gate._exponent_neg = value.canonicalize_half_turns(exponent_neg)

    @property
    def exponent_pos(self):
        """The positive exponent."""
        return self.gate.exponent_pos

    @exponent_pos.setter  # type: ignore
    @deprecated(
        deadline="v0.15",
        fix="The mutators of this class are deprecated, instantiate a new object instead.",
    )
    def exponent_pos(self, exponent_pos):
        """Sets the positive exponent."""
        # coverage: ignore
        self.gate._exponent_pos = value.canonicalize_half_turns(exponent_pos)

    @property
    def pauli_string(self):
        """The underlying pauli string."""
        return self._pauli_string

    @pauli_string.setter  # type: ignore
    @deprecated(
        deadline="v0.15",
        fix="The mutators of this class are deprecated, instantiate a new object instead.",
    )
    def pauli_string(self, pauli_string):
        """Sets the underlying pauli string."""
        # coverage: ignore
        self._pauli_string = pauli_string
        self.gate._dense_pauli_string = pauli_string.dense(pauli_string.qubits)
        super()._qubits = pauli_string.qubits

    @property
    def exponent_relative(self) -> Union[int, float, sympy.Basic]:
        """The relative exponent between negative and positive exponents."""
        return self.gate.exponent_relative

    def _value_equality_values_(self):
        return (
            self.pauli_string,
            self.exponent_neg,
            self.exponent_pos,
        )

    def equal_up_to_global_phase(self, other):
        """Checks equality of two PauliStringPhasors, up to global phase."""
        if isinstance(other, PauliStringPhasor):
            rel1 = self.exponent_relative
            rel2 = other.exponent_relative
            return rel1 == rel2 and self.pauli_string == other.pauli_string
        return False

    def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]):
        """Maps the qubits inside the PauliString."""
        return PauliStringPhasor(
            self.pauli_string.map_qubits(qubit_map),
            exponent_neg=self.exponent_neg,
            exponent_pos=self.exponent_pos,
        )

    def can_merge_with(self, op: 'PauliStringPhasor') -> bool:
        """Checks whether the underlying PauliStrings can be merged."""
        return self.pauli_string.equal_up_to_coefficient(op.pauli_string)

    def merged_with(self, op: 'PauliStringPhasor') -> 'PauliStringPhasor':
        """Merges two PauliStringPhasors."""
        if not self.can_merge_with(op):
            raise ValueError(f'Cannot merge operations: {self}, {op}')
        pp = self.exponent_pos + op.exponent_pos
        pn = self.exponent_neg + op.exponent_neg
        return PauliStringPhasor(self.pauli_string, exponent_pos=pp, exponent_neg=pn)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        qubits = self.qubits if args.known_qubits is None else args.known_qubits
        syms = tuple(f'[{self.pauli_string[qubit]}]' for qubit in qubits)
        return protocols.CircuitDiagramInfo(wire_symbols=syms, exponent=self.exponent_relative)

    def pass_operations_over(
        self, ops: Iterable[raw_types.Operation], after_to_before: bool = False
    ) -> 'PauliStringPhasor':
        """Determines how the Pauli phasor changes when conjugated by Cliffords.

        The output and input pauli phasors are related by a circuit equivalence.
        In particular, this circuit:

            ───ops───INPUT_PAULI_PHASOR───

        will be equivalent to this circuit:

            ───OUTPUT_PAULI_PHASOR───ops───

        up to global phase (assuming `after_to_before` is not set).

        If ops together have matrix C, the Pauli string has matrix P, and the
        output Pauli string has matrix P', then P' == C^-1 P C up to
        global phase.

        Setting `after_to_before` inverts the relationship, so that the output
        is the input and the input is the output. Equivalently, it inverts C.

        Args:
            ops: The operations to move over the string.
            after_to_before: Determines whether the operations start after the
                pauli string, instead of before (and so are moving in the
                opposite direction).
        """
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

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['pauli_string', 'exponent_neg', 'exponent_pos'])


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
                "Given DensePauliString doesn't have +1 and -1 eigenvalues. "
                "dense_pauli_string.coefficient must be 1 or -1."
            )

        self._dense_pauli_string = dense_pauli_string
        self._exponent_neg = value.canonicalize_half_turns(exponent_neg)
        self._exponent_pos = value.canonicalize_half_turns(exponent_pos)

    @property
    def exponent_relative(self) -> Union[int, float, sympy.Basic]:
        """The relative exponent between negative and positive exponents."""
        return value.canonicalize_half_turns(self.exponent_neg - self.exponent_pos)

    @property
    def exponent_neg(self):
        """The negative exponent."""
        return self._exponent_neg

    @property
    def exponent_pos(self):
        """The positive exponent."""
        return self._exponent_pos

    @property
    def dense_pauli_string(self):
        """The underlying DensePauliString."""
        return self._dense_pauli_string

    def _value_equality_values_(self):
        return (
            self.dense_pauli_string,
            self.exponent_neg,
            self.exponent_pos,
        )

    def equal_up_to_global_phase(self, other):
        """Checks equality of two PauliStringPhasors, up to global phase."""
        if isinstance(other, PauliStringPhasorGate):
            rel1 = self.exponent_relative
            rel2 = other.exponent_relative
            return rel1 == rel2 and self.dense_pauli_string == other.dense_pauli_string
        return False

    def __pow__(self, exponent: Union[float, sympy.Symbol]) -> 'PauliStringPhasorGate':
        pn = protocols.mul(self.exponent_neg, exponent, None)
        pp = protocols.mul(self.exponent_pos, exponent, None)
        if pn is None or pp is None:
            return NotImplemented
        return PauliStringPhasorGate(self.dense_pauli_string, exponent_neg=pn, exponent_pos=pp)

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _to_z_basis_ops(self, qubits: Sequence['cirq.Qid']) -> Iterator[raw_types.Operation]:
        """Returns operations to convert the qubits to the computational basis."""
        return self.dense_pauli_string.on(*qubits).to_z_basis_ops()

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        if len(self.dense_pauli_string) <= 0:
            return
        any_qubit = qubits[0]
        to_z_ops = op_tree.freeze_op_tree(self._to_z_basis_ops(qubits))
        xor_decomp = tuple(xor_nonlocal_decompose(qubits, any_qubit))
        yield to_z_ops
        yield xor_decomp

        if self.exponent_neg:
            yield pauli_gates.Z(any_qubit) ** self.exponent_neg
        if self.exponent_pos:
            yield pauli_gates.X(any_qubit)
            yield pauli_gates.Z(any_qubit) ** self.exponent_pos
            yield pauli_gates.X(any_qubit)

        yield protocols.inverse(xor_decomp)
        yield protocols.inverse(to_z_ops)

    def _trace_distance_bound_(self) -> float:
        if len(self.dense_pauli_string) == 0:
            return 0.0
        return protocols.trace_distance_bound(pauli_gates.Z**self.exponent_relative)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.exponent_neg) or protocols.is_parameterized(
            self.exponent_pos
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.exponent_neg) | protocols.parameter_names(
            self.exponent_pos
        )

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'PauliStringPhasorGate':
        return PauliStringPhasorGate(
            self.dense_pauli_string,
            exponent_neg=resolver.value_of(self.exponent_neg, recursive),
            exponent_pos=resolver.value_of(self.exponent_pos, recursive),
        )

    def __repr__(self) -> str:
        return (
            f'cirq.PauliStringPhasorGate({self.dense_pauli_string!r}, '
            f'exponent_neg={proper_repr(self.exponent_neg)}, '
            f'exponent_pos={proper_repr(self.exponent_pos)})'
        )

    def __str__(self) -> str:
        if self.exponent_pos == -self.exponent_neg:
            sign = '-' if self.exponent_pos < 0 else ''
            exponent = str(abs(self.exponent_pos))
            return f'exp({sign}iπ{exponent}*{self.dense_pauli_string})'
        return f'({self.dense_pauli_string})**{self.exponent_relative}'

    def num_qubits(self) -> int:
        """The number of qubits for the gate."""
        return len(self.dense_pauli_string)

    def on(self, *qubits: 'cirq.Qid') -> 'cirq.PauliStringPhasor':
        """Creates a PauliStringPhasor on the qubits."""
        return PauliStringPhasor(
            self.dense_pauli_string.on(*qubits),
            exponent_pos=self.exponent_pos,
            exponent_neg=self.exponent_neg,
        )

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(
            self, ['dense_pauli_string', 'exponent_neg', 'exponent_pos']
        )


def xor_nonlocal_decompose(
    qubits: Iterable[raw_types.Qid], onto_qubit: 'cirq.Qid'
) -> Iterable[raw_types.Operation]:
    """Decomposition ignores connectivity."""
    for qubit in qubits:
        if qubit != onto_qubit:
            yield common_gates.CNOT(qubit, onto_qubit)

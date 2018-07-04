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

"""Marker classes for indicating which additional features gates support.

For example: some gates are reversible, some have known matrices, etc.
"""

from typing import Optional, Sequence, Tuple, Type, Union, Iterable, TypeVar

import numpy as np

from cirq import abc, value
from cirq.ops import op_tree
from cirq.ops import raw_types
from cirq.study import ParamResolver


class ReversibleEffect(metaclass=abc.ABCMeta):
    """A gate whose effect can be undone in a known way."""

    @abc.abstractmethod
    def inverse(self) -> 'ReversibleEffect':
        """Returns a gate with an exactly opposite effect."""


TSelf_ExtrapolatableEffect = TypeVar('TSelf_ExtrapolatableEffect',
                                     bound='ExtrapolatableEffect')


class ExtrapolatableEffect(ReversibleEffect,
                           metaclass=abc.ABCMeta):
    """A gate whose effect can be continuously scaled up/down/negated."""

    @abc.abstractmethod
    def extrapolate_effect(self: TSelf_ExtrapolatableEffect, factor: float
                           ) -> TSelf_ExtrapolatableEffect:
        """Augments, diminishes, or reverses the effect of the receiving gate.

        Args:
            factor: The amount to scale the gate's effect by.

        Returns:
            A gate equivalent to applying the receiving gate 'factor' times.
        """

    def __pow__(self: TSelf_ExtrapolatableEffect, power: float
                ) -> TSelf_ExtrapolatableEffect:
        """Extrapolates the effect of the gate.

        Note that there are cases where (G**a)**b != G**(a*b). For example,
        start with a 90 degree rotation then cube it then raise it to a
        non-integer power such as 3/2. Assuming that rotations are always
        normalized into the range (-180, 180], note that:

            ((rot 90)**3)**1.5 = (rot 270)**1.5 = (rot -90)**1.5 = rot -135

        but

            (rot 90)**(3*1.5) = (rot 90)**4.5 = rot 405 = rot 35

        Because normalization discards the winding number.

        Args:
          power: The extrapolation factor.

        Returns:
          A gate with the extrapolated effect.
        """
        return self.extrapolate_effect(power)

    def inverse(self: TSelf_ExtrapolatableEffect) -> TSelf_ExtrapolatableEffect:
        return self.extrapolate_effect(-1)


class CompositeGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate with a known decomposition into multiple simpler gates."""

    @abc.abstractmethod
    def default_decompose(
            self, qubits: Sequence[raw_types.QubitId]) -> op_tree.OP_TREE:
        """Yields operations for performing this gate on the given qubits.

        Args:
            qubits: The qubits the operation should be applied to.
        """

    @classmethod
    def from_gates(cls: Type,
        gates: Union[Sequence[raw_types.Gate], Sequence[
            Tuple[raw_types.Gate, Tuple[int]]]]) -> 'CompositeGate':
        """Returns a CompositeGate which decomposes into the given gates.

        Args:
            gates: Either a sequence of gates or a sequences of (gate, indices)
                tuples, where indices is a tuple of qubit indices (ints). The
                first is used when decomposing a gate into a series of gates
                that all act on the same number of qubits. The second is used
                when decomposing a gate into a series of gates that may act on
                differing number of qubits. In this later case the indices
                is a tuple of qubit indices, describing which qubit the gate
                acts on.

        Returns:
            A CompositeGate with a default_decompose that applies the
            given gates in sequence.
        """
        return _CompositeGateImpl(gates)


class _CompositeGateImpl(CompositeGate):
    """Implementation of CompositeGate which uses specific sequence of gates."""

    def __init__(self, gates: Union[Sequence[raw_types.Gate], Sequence[
        Tuple[raw_types.Gate, Tuple[int]]]]) -> None:
        self.gates = gates

    def default_decompose(
        self, qubits: Sequence[raw_types.QubitId]) -> op_tree.OP_TREE:
        decomposition = []
        for x in self.gates:
            if isinstance(x, raw_types.Gate):
                decomposition.append(x(*qubits))
            else:
                gate, indices = x
                decomposition.append(gate(*map(qubits.__getitem__, indices)))
        return decomposition


class KnownMatrixGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate whose constant non-parameterized effect has a known matrix."""

    @abc.abstractmethod
    def matrix(self) -> np.ndarray:
        """The unitary matrix of the operation this gate applies."""


class TextDiagrammableGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate which can be nicely represented in a text diagram."""

    # noinspection PyMethodMayBeStatic
    def text_diagram_exponent(self) -> Union[float, value.Symbol]:
        """The exponent to modify the gate symbol with. 1 means no modifier."""
        return 1

    @abc.abstractmethod
    def text_diagram_wire_symbols(self,
                                  qubit_count: Optional[int] = None,
                                  use_unicode_characters: bool = True,
                                  precision: Optional[int] = 3
                                  ) -> Tuple[str, ...]:
        """The symbols that should be shown on the gate's qubits (in order).

        If the gate always acts on the same number of qubits, then the size
        of the returned tuple should be equal to this number of qubits.
        If the gate acts on a variable number of qubits, then a single
        symbol should be used, and this will be repeated across the operation.
        It is an error to have more than a single symbol in the case that
        the gate acts on a variable number of qubits.

        Args:
            qubit_count: The number of qubits the gate is being applied to, if
                this information is known by the caller.
            use_unicode_characters: If true, the wire symbols are permitted to
                include unicode characters (as long as they work well in fixed
                width fonts). If false, use only ascii characters. ASCII is
                preferred in cases where UTF8 support is done poorly, or where
                the fixed-width font being used to show the diagrams does not
                properly handle unicode characters.
            precision: The number of digits after the decimal to show in the
                text diagram.

        Returns:
             A tuple containing symbols to place on each of the qubit wires
             touched by the gate.
        """


TSelf_PhaseableGate = TypeVar('TSelf_PhaseableGate', bound='PhaseableGate')


class PhaseableGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate whose effect can be phased around the Z axis of target qubits."""

    @abc.abstractmethod
    def phase_by(self: TSelf_PhaseableGate,
                 phase_turns: float,
                 qubit_index: int) -> TSelf_PhaseableGate:
        """Returns a phased version of the effect.

        For example, an X gate phased by 90 degrees would be a Y gate.

        Args:
            phase_turns: The amount to phase the gate, in fractions of a whole
                turn.
            qubit_index: The index of the target qubit the phasing applies to.

        Returns:
            The phased gate.
        """


class BoundedEffect(metaclass=abc.ABCMeta):
    """An effect with known bounds on how easy it is to detect.

    Used when deciding whether or not an operation is negligible. For example,
    the trace distance between the states before and after a Z**0.00000001
    operation is very close to 0, so it would typically be considered
    negligible.
    """

    @abc.abstractmethod
    def trace_distance_bound(self) -> float:
        """A maximum on the trace distance between this effect's input/output.

        Generally this method is used when deciding whether to keep gates, so
        only the behavior near 0 is important. Approximations that overestimate
        the maximum trace distance are permitted. Even ones that exceed 1.
        Underestimates are not permitted.
        """


class SingleQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly one qubit."""

    def validate_args(self, qubits):
        if len(qubits) != 1:
            raise ValueError(
                'Single-qubit gate applied to multiple qubits: {}({})'.
                format(self, qubits))

    def on_each(self, targets: Iterable[raw_types.QubitId]) -> op_tree.OP_TREE:
        """Returns a list of operations apply this gate to each of the targets.

        Args:
            targets: The qubits to apply this gate to.

        Returns:
            Operations applying this gate to the target qubits.
        """
        return [self.on(target) for target in targets]


class TwoQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly two qubits."""

    def validate_args(self, qubits):
        if len(qubits) != 2:
            raise ValueError(
                'Two-qubit gate not applied to two qubits: {}({})'.
                format(self, qubits))


class ThreeQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly three qubits."""

    def validate_args(self, qubits):
        if len(qubits) != 3:
            raise ValueError(
                'Three-qubit gate not applied to three qubits: {}({})'.
                format(self, qubits))


TSelf_ParameterizableEffect = TypeVar('TSelf_ParameterizableEffect',
                                      bound='ParameterizableEffect')


class ParameterizableEffect(metaclass=abc.ABCMeta):
    """An effect that can be parameterized by Symbols."""

    @abc.abstractmethod
    def is_parameterized(self) -> bool:
        """Whether the effect is parameterized.
        
        Returns True if the gate has any unresolved Symbols and False otherwise.
        """

    @abc.abstractmethod
    def with_parameters_resolved_by(self: TSelf_ParameterizableEffect,
                                    param_resolver: ParamResolver
                                    ) -> TSelf_ParameterizableEffect:
        """Resolve the parameters in the effect.

        Returns a gate or operation of the same type, but with all Symbols
        replaced with floats according to the given ParamResolver.
        """

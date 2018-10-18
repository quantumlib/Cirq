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

from typing import (
    Any, Dict, Optional, Sequence, Tuple, Iterable, TypeVar, Union,
)

import abc
import string

from cirq import value
from cirq.ops import op_tree, raw_types


class InterchangeableQubitsGate(metaclass=abc.ABCMeta):
    """Indicates operations should be equal under some qubit permutations."""

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        """Returns a key that differs between non-interchangeable qubits."""
        return 0



TSelf_ExtrapolatableEffect = TypeVar('TSelf_ExtrapolatableEffect',
                                     bound='ExtrapolatableEffect')


class ExtrapolatableEffect(metaclass=abc.ABCMeta):
    """A gate whose effect can be continuously scaled up/down/negated."""

    @abc.abstractmethod
    def extrapolate_effect(self: TSelf_ExtrapolatableEffect,
                           factor: Union[float, value.Symbol]
                           ) -> TSelf_ExtrapolatableEffect:
        """Augments, diminishes, or reverses the effect of the receiving gate.

        Args:
            factor: The amount to scale the gate's effect by.

        Returns:
            A gate equivalent to applying the receiving gate 'factor' times.
        """

    def __pow__(self: TSelf_ExtrapolatableEffect,
                power: Union[float, value.Symbol]
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


class CompositeOperation(metaclass=abc.ABCMeta):
    """An operation with a known decomposition into simpler operations."""

    @abc.abstractmethod
    def default_decompose(self) -> op_tree.OP_TREE:
        """Yields simpler operations for performing the receiving operation."""


class CompositeGate(metaclass=abc.ABCMeta):
    """A gate with a known decomposition into simpler gates."""

    @abc.abstractmethod
    def default_decompose(
            self, qubits: Sequence[raw_types.QubitId]) -> op_tree.OP_TREE:
        """Yields operations for performing this gate on the given qubits.

        Args:
            qubits: The qubits the gate should be applied to.
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



class QasmOutputArgs(string.Formatter):
    """
    Attributes:
        precision: The number of digits after the decimal to show for numbers in
            the text diagram.
        version: The QASM version to output.  QasmConvertibleGate/Operation may
            return different text depending on version.
        qubit_id_map: A dictionary mapping qubits to qreg QASM identifiers.
        meas_key_id_map: A dictionary mapping measurement keys to creg QASM
            identifiers.
    """
    def __init__(self,
                 precision: int = 10,
                 version: str = '2.0',
                 qubit_id_map: Dict[raw_types.QubitId, str] = None,
                 meas_key_id_map: Dict[str, str] = None,
                 ) -> None:
        self.precision = precision
        self.version = version
        self.qubit_id_map = {} if qubit_id_map is None else qubit_id_map
        self.meas_key_id_map = ({} if meas_key_id_map is None
                                   else meas_key_id_map)

    def format_field(self, value: Any, spec: str) -> str:
        """Method of string.Formatter that specifies the output of format()."""
        if isinstance(value, float):
            value = round(value, self.precision)
            if spec == 'half_turns':
                value = 'pi*{}'.format(value) if value != 0 else '0'
                spec = ''
        elif isinstance(value, raw_types.QubitId):
            value = self.qubit_id_map[value]
        elif isinstance(value, str) and spec == 'meas':
            value = self.meas_key_id_map[value]
            spec = ''
        return super().format_field(value, spec)

    def validate_version(self, *supported_versions: str) -> None:
        if self.version not in supported_versions:
            raise ValueError('QASM version {} output is not supported.'.format(
                                self.version))


class QasmConvertibleGate(metaclass=abc.ABCMeta):
    """A gate that knows its representation in QASM."""
    @abc.abstractmethod
    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: QasmOutputArgs) -> Optional[str]:
        """Returns lines of QASM output representing the gate on the given
        qubits or None if a simple conversion is not possible.
        """


class QasmConvertibleOperation(metaclass=abc.ABCMeta):
    """An operation that knows its representation in QASM."""
    @abc.abstractmethod
    def known_qasm_output(self, args: QasmOutputArgs) -> Optional[str]:
        """Returns lines of QASM output representing the operation or None if a
        simple conversion is not possible."""

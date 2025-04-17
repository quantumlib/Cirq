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
"""Helper class for implementing classical arithmetic operations."""

from __future__ import annotations

import abc
import itertools
from typing import cast, Iterable, List, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
from typing_extensions import Self

from cirq.ops.raw_types import Gate

if TYPE_CHECKING:
    import cirq


class ArithmeticGate(Gate, metaclass=abc.ABCMeta):
    r"""A helper gate for implementing reversible classical arithmetic.

    Child classes must override the `registers`, `with_registers`, and `apply`
    methods.

    This class handles the details of ensuring that the scaling of implementing
    the gate is O(2^n) instead of O(4^n) where n is the number of qubits
    being acted on, by implementing an `_apply_unitary_` function in terms of
    the registers and the apply function of the child class.

    Examples:

    >>> class Add(cirq.ArithmeticGate):
    ...     def __init__(
    ...         self,
    ...         target_register: '[int, Sequence[int]]',
    ...         input_register: 'Union[int, Sequence[int]]',
    ...     ):
    ...         self.target_register = target_register
    ...         self.input_register = input_register
    ...
    ...     def registers(self) -> 'Sequence[Union[int, Sequence[int]]]':
    ...         return self.target_register, self.input_register
    ...
    ...     def with_registers(
    ...         self, *new_registers: 'Union[int, Sequence[int]]'
    ...     ) -> 'Add':
    ...         return Add(*new_registers)
    ...
    ...     def apply(self, *register_values: int) -> 'Union[int, Iterable[int]]':
    ...         return sum(register_values)
    >>> cirq.unitary(
    ...     Add(target_register=[2, 2],
    ...         input_register=1).on(*cirq.LineQubit.range(2))
    ... ).astype(np.int32)
    array([[0, 0, 0, 1],
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0]], dtype=int32)
    >>> c = cirq.Circuit(
    ...    cirq.X(cirq.LineQubit(3)),
    ...    cirq.X(cirq.LineQubit(2)),
    ...    cirq.X(cirq.LineQubit(6)),
    ...    cirq.measure(*cirq.LineQubit.range(4, 8), key='before_in'),
    ...    cirq.measure(*cirq.LineQubit.range(4), key='before_out'),
    ...
    ...    Add(target_register=[2] * 4,
    ...        input_register=[2] * 4).on(*cirq.LineQubit.range(8)),
    ...
    ...    cirq.measure(*cirq.LineQubit.range(4, 8), key='after_in'),
    ...    cirq.measure(*cirq.LineQubit.range(4), key='after_out'),
    ... )
    >>> cirq.sample(c).data
       before_in  before_out  after_in  after_out
    0          2           3         2          5

    """

    @abc.abstractmethod
    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        """The data acted upon by the arithmetic gate.

        Each register in the list can either be a classical constant (an `int`),
        or else a list of qubit/qudit dimensions. Registers that are set to a
        classical constant must not be mutated by the arithmetic gate
        (their value must remain fixed when passed to `apply`).

        Registers are big endian. The first qubit is the most significant, the
        last qubit is the 1s qubit, the before last qubit is the 2s qubit, etc.

        Returns:
            A list of constants and qubit groups that the gate will act upon.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> Self:
        """Returns the same fate targeting different registers.

        Args:
            *new_registers: The new values that should be returned by the
                `registers` method.

        Returns:
            An instance of the same kind of gate, but acting on different
            registers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        """Returns the result of the gate operating on classical values.

        For example, an addition takes two values (the target and the source),
        adds the source into the target, then returns the target and source
        as the new register values.

        The `apply` method is permitted to be sloppy in three ways:

        1. The `apply` method is permitted to return values that have more bits
            than the registers they will be stored into. The extra bits are
            simply dropped. For example, if the value 5 is returned for a 2
            qubit register then 5 % 2**2 = 1 will be used instead. Negative
            values are also permitted. For example, for a 3 qubit register the
            value -2 becomes -2 % 2**3 = 6.
        2. When the value of the last `k` registers is not changed by the
            gate, the `apply` method is permitted to omit these values
            from the result. That is to say, when the length of the output is
            less than the length of the input, it is padded up to the intended
            length by copying from the same position in the input.
        3. When only the first register's value changes, the `apply` method is
            permitted to return an `int` instead of a sequence of ints.

        The `apply` method *must* be reversible. Otherwise the gate will
        not be unitary, and incorrect behavior will result.

        Examples:

            A fully detailed adder:

            ```
            def apply(self, target, offset):
                return (target + offset) % 2**len(self.target_register), offset
            ```

            The same adder, with less boilerplate due to the details being
            handled by the `ArithmeticGate` class:

            ```
            def apply(self, target, offset):
                return target + offset
            ```
        """
        raise NotImplementedError()

    def _qid_shape_(self) -> Tuple[int, ...]:
        shape = []
        for r in self.registers():
            if isinstance(r, Sequence):
                for i in r:
                    shape.append(i)
        return tuple(shape)

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
        registers = self.registers()
        input_ranges: List[Sequence[int]] = []
        shape: List[int] = []
        overflow_sizes: List[int] = []
        for register in registers:
            if isinstance(register, int):
                input_ranges.append([register])
                shape.append(1)
                overflow_sizes.append(register + 1)
            else:
                size = int(np.prod([dim for dim in register], dtype=np.int64).item())
                shape.append(size)
                input_ranges.append(range(size))
                overflow_sizes.append(size)

        leftover = args.target_tensor.size // np.prod(shape, dtype=np.int64).item()
        new_shape = (*shape, leftover)

        transposed_args = args.with_axes_transposed_to_start()
        src = transposed_args.target_tensor.reshape(new_shape)
        dst = transposed_args.available_buffer.reshape(new_shape)
        for input_seq in itertools.product(*input_ranges):
            output = self.apply(*input_seq)

            # Wrap into list.
            inputs: List[int] = list(input_seq)
            outputs: List[int] = [output] if isinstance(output, int) else list(output)

            # Omitted tail values default to the corresponding input value.
            if len(outputs) < len(inputs):
                outputs += inputs[len(outputs) - len(inputs) :]
            # Get indices into range.
            for i in range(len(outputs)):
                if isinstance(registers[i], int):
                    if outputs[i] != registers[i]:
                        raise ValueError(
                            _describe_bad_arithmetic_changed_const(
                                self.registers(), inputs, outputs
                            )
                        )
                    # Classical constants go to zero on a unit axe.
                    outputs[i] = 0
                    inputs[i] = 0
                else:
                    # Quantum values get wrapped into range.
                    outputs[i] %= overflow_sizes[i]

            # Copy amplitude to new location.
            cast(List[Union[int, slice]], outputs).append(slice(None))
            cast(List[Union[int, slice]], inputs).append(slice(None))
            dst[tuple(outputs)] = src[tuple(inputs)]

        # In case the reshaped arrays were copies instead of views.
        dst.shape = transposed_args.available_buffer.shape
        transposed_args.target_tensor[...] = dst

        return args.target_tensor


def _describe_bad_arithmetic_changed_const(
    registers: Sequence[Union[int, Sequence[Union[cirq.Qid, int]]]],
    inputs: List[int],
    outputs: List[int],
) -> str:
    from cirq.circuits import TextDiagramDrawer

    drawer = TextDiagramDrawer()
    drawer.write(0, 0, 'Register Data')
    drawer.write(1, 0, 'Register Type')
    drawer.write(2, 0, 'Input Value')
    drawer.write(3, 0, 'Output Value')
    for i in range(len(registers)):
        drawer.write(0, i + 1, str(registers[i]))
        drawer.write(1, i + 1, 'constant' if isinstance(registers[i], int) else 'qureg')
        drawer.write(2, i + 1, str(inputs[i]))
        drawer.write(3, i + 1, str(outputs[i]))
    return (
        "A register cannot be set to an int (a classical constant) unless its "
        "value is not affected by the gate.\n"
        "\nExample case where a constant changed:\n"
        + drawer.render(horizontal_spacing=1, vertical_spacing=0)
    )

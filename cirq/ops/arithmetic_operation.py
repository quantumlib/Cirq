# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper class for implementing classical arithmetic operations."""

import abc
import itertools
from typing import Union, Iterable, List, Sequence, cast, TypeVar, TYPE_CHECKING

import numpy as np

from cirq.ops.raw_types import Operation

if TYPE_CHECKING:
    import cirq


TSelf = TypeVar('TSelf', bound='ArithmeticOperation')


class ArithmeticOperation(Operation, metaclass=abc.ABCMeta):
    """A helper class for implementing reversible classical arithmetic.

    Child classes must override the `registers`, `with_registers`, and `apply`
    methods.

    This class handles the details of ensuring that the scaling of implementing
    the operation is O(2^n) instead of O(4^n) where n is the number of qubits
    being acted on, by implementing an `_apply_unitary_` function in terms of
    the registers and the apply function of the child class. It also handles the
    boilerplate of implementing the `qubits` and `with_qubits` methods.

    Examples:
    ```

        >>> class Add(cirq.ArithmeticOperation):
        ...     def __init__(self, target_register, input_register):
        ...         self.target_register = target_register
        ...         self.input_register = input_register
        ...
        ...     def registers(self):
        ...         return self.target_register, self.input_register
        ...
        ...     def with_registers(self, *new_registers):
        ...         return Add(*new_registers)
        ...
        ...     def apply(self, target_value, input_value):
        ...         return target_value + input_value
        >>> cirq.unitary(
        ...     Add(target_register=cirq.LineQubit.range(2),
        ...         input_register=1)
        ... ).astype(np.int32)
        array([[0, 0, 0, 1],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=int32)
        >>> c = cirq.Circuit(
        ...    cirq.X(cirq.LineQubit(3)),
        ...    cirq.X(cirq.LineQubit(2)),
        ...    cirq.X(cirq.LineQubit(6)),
        ...    cirq.measure(*cirq.LineQubit.range(4, 8), key='before:in'),
        ...    cirq.measure(*cirq.LineQubit.range(4), key='before:out'),
        ...
        ...    Add(target_register=cirq.LineQubit.range(4),
        ...        input_register=cirq.LineQubit.range(4, 8)),
        ...
        ...    cirq.measure(*cirq.LineQubit.range(4, 8), key='after:in'),
        ...    cirq.measure(*cirq.LineQubit.range(4), key='after:out'),
        ... )
        >>> cirq.sample(c).data
           before:in  before:out  after:in  after:out
        0          2           3         2          5

    ```
    """

    @abc.abstractmethod
    def registers(self) -> Sequence[Union[int, Sequence['cirq.Qid']]]:
        """The data acted upon by the arithmetic operation.

        Each register in the list can either be a classical constant (an `int`),
        or else a list of qubits/qudits (a `List[cirq.Qid]`). Registers that
        are set to a classical constant must not be mutated by the arithmetic
        operation (their value must remain fixed when passed to `apply`).

        Registers are big endian. The first qubit is the most significant, the
        last qubit is the 1s qubit, the before last qubit is the 2s qubit, etc.

        Returns:
            A list of constants and qubit groups that the operation will act
            upon.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def with_registers(self: TSelf,
                       *new_registers: Union[int, Sequence['cirq.Qid']]
                      ) -> TSelf:
        """Returns the same operation targeting different registers.

        Args:
            new_registers: The new values that should be returned by the
                `registers` method.

        Returns:
            An instance of the same kind of operation, but acting on different
            registers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        """Returns the result of the operation operating on classical values.

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
            operation, the `apply` method is permitted to omit these values
            from the result. That is to say, when the length of the output is
            less than the length of the input, it is padded up to the intended
            length by copying from the same position in the input.
        3. When only the first register's value changes, the `apply` method is
            permitted to return an `int` instead of a sequence of ints.

        The `apply` method *must* be reversible. Otherwise the operation will
        not be unitary, and incorrect behavior will result.

        Examples:

            A fully detailed adder:

            ```
            def apply(self, target, offset):
                return (target + offset) % 2**len(self.target_register), offset
            ```

            The same adder, with less boilerplate due to the details being
            handled by the `ArithmeticOperation` class:

            ```
            def apply(self, target, offset):
                return target + offset
            ```
        """
        raise NotImplementedError()

    @property
    def qubits(self):
        return tuple(qubit for register in self.registers()
                     if not isinstance(register, int) for qubit in register)

    def with_qubits(self: TSelf, *new_qubits: 'cirq.Qid') -> TSelf:
        new_registers: List[Union[int, Sequence['cirq.Qid']]] = []
        qs = iter(new_qubits)
        for register in self.registers():
            if isinstance(register, int):
                new_registers.append(register)
            else:
                new_registers.append([next(qs) for _ in register])
        return self.with_registers(*new_registers)

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        registers = self.registers()
        input_ranges: List[Sequence[int]] = []
        shape = []
        overflow_sizes = []
        for register in registers:
            if isinstance(register, int):
                input_ranges.append([register])
                shape.append(1)
                overflow_sizes.append(register + 1)
            else:
                size = int(np.product([q.dimension for q in register]).item())
                shape.append(size)
                input_ranges.append(range(size))
                overflow_sizes.append(size)

        leftover = args.target_tensor.size // np.product(shape).item()
        new_shape = (*shape, leftover)

        transposed_args = args.with_axes_transposed_to_start()
        src = transposed_args.target_tensor.reshape(new_shape)
        dst = transposed_args.available_buffer.reshape(new_shape)
        for input_seq in itertools.product(*input_ranges):
            output = self.apply(*input_seq)

            # Wrap into list.
            inputs: List[int] = list(input_seq)
            outputs: List[int] = ([output]
                                  if isinstance(output, int) else list(output))

            # Omitted tail values default to the corresponding input value.
            if len(outputs) < len(inputs):
                outputs += inputs[len(outputs) - len(inputs):]
            # Get indices into range.
            for i in range(len(outputs)):
                if isinstance(registers[i], int):
                    if outputs[i] != registers[i]:
                        raise ValueError(
                            _describe_bad_arithmetic_changed_const(
                                self.registers(), inputs, outputs))
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
        registers: Sequence[Union[int, Sequence['cirq.Qid']]],
        inputs: List[int], outputs: List[int]) -> str:
    from cirq.circuits import TextDiagramDrawer
    drawer = TextDiagramDrawer()
    drawer.write(0, 0, 'Register Data')
    drawer.write(1, 0, 'Register Type')
    drawer.write(2, 0, 'Input Value')
    drawer.write(3, 0, 'Output Value')
    for i in range(len(registers)):
        drawer.write(0, i + 1, str(registers[i]))
        drawer.write(1, i + 1,
                     'constant' if isinstance(registers[i], int) else 'qureg')
        drawer.write(2, i + 1, str(inputs[i]))
        drawer.write(3, i + 1, str(outputs[i]))
    return (
        "A register cannot be set to an int (a classical constant) unless its "
        "value is not affected by the operation.\n"
        "\nExample case where a constant changed:\n" +
        drawer.render(horizontal_spacing=1, vertical_spacing=0))

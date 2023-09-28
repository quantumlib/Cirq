# Copyright 2023 The Cirq Developers
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

import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray

import attr
import cirq
import numpy as np


class Side(enum.Flag):
    """Denote LEFT, RIGHT, or THRU signature.

    LEFT signature serve as input lines (only) to the Gate. RIGHT signature are output
    lines (only) from the Gate. THRU signature are both input and output.

    Traditional unitary operations will have THRU signature that operate on a collection of
    qubits which are then made available to following operations. RIGHT and LEFT signature
    imply allocation, deallocation, or reshaping of the signature.
    """

    LEFT = enum.auto()
    RIGHT = enum.auto()
    THRU = LEFT | RIGHT


@attr.frozen
class Register:
    """A quantum register used to define the input/output API of a `cirq_ft.GateWithRegister`

    Attributes:
        name: The string name of the register
        bitsize: The number of (qu)bits in the register.
        shape: A tuple of integer dimensions to declare a multidimensional register. The
            total number of bits is the product of entries in this tuple times `bitsize`.
        side: Whether this is a left, right, or thru register. See the documentation for `Side`
            for more information.
    """

    name: str
    bitsize: int = attr.field()
    shape: Tuple[int, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )
    side: Side = Side.THRU

    @bitsize.validator
    def bitsize_validator(self, attribute, value):
        if value <= 0:
            raise ValueError(f"Bitsize for {self=} must be a positive integer. Found {value}.")

    def all_idxs(self) -> Iterable[Tuple[int, ...]]:
        """Iterate over all possible indices of a multidimensional register."""
        yield from itertools.product(*[range(sh) for sh in self.shape])

    def total_bits(self) -> int:
        """The total number of bits in this register.

        This is the product of each of the dimensions in `shape`.
        """
        return self.bitsize * int(np.prod(self.shape))

    def __repr__(self):
        return (
            f'cirq_ft.Register('
            f'name="{self.name}", '
            f'bitsize={self.bitsize}, '
            f'shape={self.shape}, '
            f'side=cirq_ft.infra.{self.side})'
        )


def total_bits(registers: Iterable[Register]) -> int:
    """Sum of `reg.total_bits()` for each register `reg` in input `signature`."""
    return sum(reg.total_bits() for reg in registers)


def split_qubits(
    registers: Iterable[Register], qubits: Sequence[cirq.Qid]
) -> Dict[str, NDArray[cirq.Qid]]:  # type: ignore[type-var]
    """Splits the flat list of qubits into a dictionary of appropriately shaped qubit arrays."""

    qubit_regs = {}
    base = 0
    for reg in registers:
        qubit_regs[reg.name] = np.array(qubits[base : base + reg.total_bits()]).reshape(
            reg.shape + (reg.bitsize,)
        )
        base += reg.total_bits()
    return qubit_regs


def merge_qubits(
    registers: Iterable[Register],
    **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid], NDArray[cirq.Qid]],
) -> List[cirq.Qid]:
    """Merges the dictionary of appropriately shaped qubit arrays into a flat list of qubits."""

    ret: List[cirq.Qid] = []
    for reg in registers:
        if reg.name not in qubit_regs:
            raise ValueError(f"All qubit registers must be present. {reg.name} not in qubit_regs")
        qubits = qubit_regs[reg.name]
        qubits = np.array([qubits] if isinstance(qubits, cirq.Qid) else qubits)
        full_shape = reg.shape + (reg.bitsize,)
        if qubits.shape != full_shape:
            raise ValueError(
                f'{reg.name} register must of shape {full_shape} but is of shape {qubits.shape}'
            )
        ret += qubits.flatten().tolist()
    return ret


def get_named_qubits(registers: Iterable[Register]) -> Dict[str, NDArray[cirq.Qid]]:
    """Returns a dictionary of appropriately shaped named qubit signature for input `signature`."""

    def _qubit_array(reg: Register):
        qubits = np.empty(reg.shape + (reg.bitsize,), dtype=object)
        for ii in reg.all_idxs():
            for j in range(reg.bitsize):
                prefix = "" if not ii else f'[{", ".join(str(i) for i in ii)}]'
                suffix = "" if reg.bitsize == 1 else f"[{j}]"
                qubits[ii + (j,)] = cirq.NamedQubit(reg.name + prefix + suffix)
        return qubits

    def _qubits_for_reg(reg: Register):
        if len(reg.shape) > 0:
            return _qubit_array(reg)

        return np.array(
            [cirq.NamedQubit(f"{reg.name}")]
            if reg.total_bits() == 1
            else cirq.NamedQubit.range(reg.total_bits(), prefix=reg.name),
            dtype=object,
        )

    return {reg.name: _qubits_for_reg(reg) for reg in registers}


class Signature:
    """An ordered collection of `cirq_ft.Register`.

    Args:
        registers: an iterable of the contained `cirq_ft.Register`.
    """

    def __init__(self, registers: Iterable[Register]):
        self._registers = tuple(registers)
        self._lefts = {r.name: r for r in self._registers if r.side & Side.LEFT}
        self._rights = {r.name: r for r in self._registers if r.side & Side.RIGHT}
        if len(set(self._lefts) | set(self._rights)) != len(self._registers):
            raise ValueError("Please provide unique register names.")

    def __repr__(self):
        return f'cirq_ft.Signature({self._registers})'

    @classmethod
    def build(cls, **registers: int) -> 'Signature':
        return cls(Register(name=k, bitsize=v) for k, v in registers.items() if v > 0)

    @overload
    def __getitem__(self, key: int) -> Register:
        pass

    @overload
    def __getitem__(self, key: slice) -> Tuple[Register, ...]:
        pass

    def __getitem__(self, key):
        return self._registers[key]

    def get_left(self, name: str) -> Register:
        """Get a left register by name."""
        return self._lefts[name]

    def get_right(self, name: str) -> Register:
        """Get a right register by name."""
        return self._rights[name]

    def __contains__(self, item: Register) -> bool:
        return item in self._registers

    def __iter__(self) -> Iterator[Register]:
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def __eq__(self, other) -> bool:
        return self._registers == other._registers

    def __hash__(self):
        return hash(self._registers)


@attr.frozen
class SelectionRegister(Register):
    """Register used to represent SELECT register for various LCU methods.

    `SelectionRegister` extends the `Register` class to store the iteration length
    corresponding to that register along with its size.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. Such (nested) coherent
    for-loops can be represented using a `Tuple[SelectionRegister, ...]` where the i'th entry
    stores the bitsize and iteration length of i'th nested for-loop.

    One useful feature when processing such nested for-loops is to flatten out a composite index,
    represented by a tuple of indices (i, j, ...), one for each selection register into a single
    integer that can be used to index a flat target register. An example of such a mapping
    function is described in Eq.45 of https://arxiv.org/abs/1805.03662. A general version of this
    mapping function can be implemented using `numpy.ravel_multi_index` and `numpy.unravel_index`.

    For example:
        1) We can flatten a 2D for-loop as follows
        >>> import numpy as np
        >>> N, M = 10, 20
        >>> flat_indices = set()
        >>> for x in range(N):
        ...     for y in range(M):
        ...         flat_idx = x * M + y
        ...         assert np.ravel_multi_index((x, y), (N, M)) == flat_idx
        ...         assert np.unravel_index(flat_idx, (N, M)) == (x, y)
        ...         flat_indices.add(flat_idx)
        >>> assert len(flat_indices) == N * M

        2) Similarly, we can flatten a 3D for-loop as follows
        >>> import numpy as np
        >>> N, M, L = 10, 20, 30
        >>> flat_indices = set()
        >>> for x in range(N):
        ...     for y in range(M):
        ...         for z in range(L):
        ...             flat_idx = x * M * L + y * L + z
        ...             assert np.ravel_multi_index((x, y, z), (N, M, L)) == flat_idx
        ...             assert np.unravel_index(flat_idx, (N, M, L)) == (x, y, z)
        ...             flat_indices.add(flat_idx)
        >>> assert len(flat_indices) == N * M * L
    """

    name: str
    bitsize: int
    iteration_length: int = attr.field()
    shape: Tuple[int, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )
    side: Side = Side.THRU

    @iteration_length.default
    def _default_iteration_length(self):
        return 2**self.bitsize

    @iteration_length.validator
    def validate_iteration_length(self, attribute, value):
        if len(self.shape) != 0:
            raise ValueError(f'Selection register {self.name} should be flat. Found {self.shape=}')
        if not (0 <= value <= 2**self.bitsize):
            raise ValueError(f'iteration length must be in range [0, 2^{self.bitsize}]')

    def __repr__(self) -> str:
        return (
            f'cirq_ft.SelectionRegister('
            f'name="{self.name}", '
            f'bitsize={self.bitsize}, '
            f'shape={self.shape}, '
            f'iteration_length={self.iteration_length})'
        )


class GateWithRegisters(cirq.Gate, metaclass=abc.ABCMeta):
    """`cirq.Gate`s extension with support for composite gates acting on multiple qubit registers.

    Though Cirq was nominally designed for circuit construction for near-term devices the core
    concept of the `cirq.Gate`, a programmatic representation of an operation on a state without
    a complete qubit address specification, can be leveraged to describe more abstract algorithmic
    primitives. To define composite gates, users derive from `cirq.Gate` and implement the
    `_decompose_` method that yields the sub-operations provided a flat list of qubits.

    This API quickly becomes inconvenient when defining operations that act on multiple qubit
    registers of variable sizes. Cirq-FT extends the `cirq.Gate` idea by introducing a new abstract
    base class `cirq_ft.GateWithRegisters` containing abstract methods `registers` and optional
    method `decompose_from_registers` that provides an overlay to the Cirq flat address API.

    As an example, in the following code snippet we use the `cirq_ft.GateWithRegisters` to
    construct a multi-target controlled swap operation:

    >>> import attr
    >>> import cirq
    >>> import cirq_ft
    >>>
    >>> @attr.frozen
    ... class MultiTargetCSwap(cirq_ft.GateWithRegisters):
    ...     bitsize: int
    ...
    ...     @property
    ...     def signature(self) -> cirq_ft.Signature:
    ...         return cirq_ft.Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)
    ...
    ...     def decompose_from_registers(self, context, ctrl, x, y) -> cirq.OP_TREE:
    ...         yield [cirq.CSWAP(*ctrl, qx, qy) for qx, qy in zip(x, y)]
    ...
    >>> op = MultiTargetCSwap(2).on_registers(
    ...     ctrl=[cirq.q('ctrl')],
    ...     x=cirq.NamedQubit.range(2, prefix='x'),
    ...     y=cirq.NamedQubit.range(2, prefix='y'),
    ... )
    >>> print(cirq.Circuit(op))
    ctrl: ───MultiTargetCSwap───
             │
    x0: ─────x──────────────────
             │
    x1: ─────x──────────────────
             │
    y0: ─────y──────────────────
             │
    y1: ─────y──────────────────"""

    @property
    @abc.abstractmethod
    def signature(self) -> Signature:
        ...

    def _num_qubits_(self) -> int:
        return total_bits(self.signature)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        return NotImplemented

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        qubit_regs = split_qubits(self.signature, qubits)
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        return self.decompose_from_registers(context=context, **qubit_regs)

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        return self._decompose_with_context_(qubits)

    def on_registers(
        self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid], NDArray[cirq.Qid]]
    ) -> cirq.Operation:
        return self.on(*merge_qubits(self.signature, **qubit_regs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Default diagram info that uses register names to name the boxes in multi-qubit gates.

        Descendants can override this method with more meaningful circuit diagram information.
        """
        wire_symbols = []
        for reg in self.signature:
            wire_symbols += [reg.name] * reg.total_bits()

        wire_symbols[0] = self.__class__.__name__
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

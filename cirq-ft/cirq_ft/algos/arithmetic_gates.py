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

from typing import Iterable, Optional, Sequence, Tuple, Union, List, Iterator
from numpy.typing import NDArray

from cirq._compat import cached_property
import attr
import cirq
from cirq_ft import infra
from cirq_ft.algos import and_gate


@attr.frozen
class LessThanGate(cirq.ArithmeticGate):
    """Applies U_a|x>|z> = |x> |z ^ (x < a)>"""

    bitsize: int
    less_than_val: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, self.less_than_val, [2]

    def with_registers(self, *new_registers) -> "LessThanGate":
        return LessThanGate(len(new_registers[0]), new_registers[1])

    def apply(self, *register_vals: int) -> Union[int, Iterable[int]]:
        input_val, less_than_val, target_register_val = register_vals
        return input_val, less_than_val, target_register_val ^ (input_val < less_than_val)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += [f'+(x < {self.less_than_val})']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover

    def __repr__(self) -> str:
        return f'cirq_ft.LessThanGate({self.bitsize}, {self.less_than_val})'

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        """Decomposes the gate into 4N And and And† operations for a T complexity of 4N.

        The decomposition proceeds from the most significant qubit -bit 0- to the least significant
        qubit while maintaining whether the qubit sequence is equal to the current prefix of the
        `_val` or not.

        The bare-bone logic is:
        1. if ith bit of `_val` is 1 then:
            - qubit sequence < `_val` iff they are equal so far and the current qubit is 0.
        2. update `are_equal`: `are_equal := are_equal and (ith bit == ith qubit).`

        This logic is implemented using $n$ `And` & `And†` operations and n+1 clean ancilla where
            - one ancilla `are_equal` contains the equality informaiton
            - ancilla[i] contain whether the qubits[:i+1] != (i+1)th prefix of `_val`
        """
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())

        qubits, target = qubits[:-1], qubits[-1]
        # Trivial case, self._val is larger than any value the registers could represent
        if self.less_than_val >= 2**self.bitsize:
            yield cirq.X(target)
            return
        adjoint = []

        (are_equal,) = context.qubit_manager.qalloc(1)

        # Initially our belief is that the numbers are equal.
        yield cirq.X(are_equal)
        adjoint.append(cirq.X(are_equal))

        # Scan from left to right.
        # `are_equal` contains whether the numbers are equal so far.
        ancilla = context.qubit_manager.qalloc(self.bitsize)
        for b, q, a in zip(
            infra.bit_tools.iter_bits(self.less_than_val, self.bitsize), qubits, ancilla
        ):
            if b:
                yield cirq.X(q)
                adjoint.append(cirq.X(q))

                # ancilla[i] = are_equal so far and (q_i != _val[i]).
                #            = equivalent to: Is the current prefix of qubits < prefix of `_val`?
                yield and_gate.And().on(q, are_equal, a)
                adjoint.append(and_gate.And(adjoint=True).on(q, are_equal, a))

                # target ^= is the current prefix of the qubit sequence < current prefix of `_val`
                yield cirq.CNOT(a, target)

                # If `a=1` (i.e. the current prefixes aren't equal) this means that
                # `are_equal` is currently = 1 and q[i] != _val[i] so we need to flip `are_equal`.
                yield cirq.CNOT(a, are_equal)
                adjoint.append(cirq.CNOT(a, are_equal))
            else:
                # ancilla[i] = are_equal so far and (q = 1).
                yield and_gate.And().on(q, are_equal, a)
                adjoint.append(and_gate.And(adjoint=True).on(q, are_equal, a))

                # if `a=1` then we need to flip `are_equal` since this means that are_equal=1,
                # b_i=0, q_i=1 => current prefixes are not equal so we need to flip `are_equal`.
                yield cirq.CNOT(a, are_equal)
                adjoint.append(cirq.CNOT(a, are_equal))

        yield from reversed(adjoint)

    def _has_unitary_(self):
        return True

    def _t_complexity_(self) -> infra.TComplexity:
        n = self.bitsize
        if self.less_than_val >= 2**n:
            return infra.TComplexity(clifford=1)
        return infra.TComplexity(
            t=4 * n, clifford=15 * n + 3 * bin(self.less_than_val).count("1") + 2
        )


@attr.frozen
class BiQubitsMixer(infra.GateWithRegisters):
    """Implements the COMPARE2 (Fig. 1) https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf

    This gates mixes the values in a way that preserves the result of comparison.
    The registers being compared are 2-qubit registers where
        x = 2*x_msb + x_lsb
        y = 2*y_msb + y_lsb
    The Gate mixes the 4 qubits so that sign(x - y) = sign(x_lsb' - y_lsb') where x_lsb' and y_lsb'
    are the final values of x_lsb' and y_lsb'.
    """  # pylint: disable=line-too-long

    adjoint: bool = False

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(x=2, y=2, ancilla=3)

    def __repr__(self) -> str:
        return f'cirq_ft.algos.BiQubitsMixer({self.adjoint})'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        x, y, ancilla = quregs['x'], quregs['y'], quregs['ancilla']
        x_msb, x_lsb = x
        y_msb, y_lsb = y

        def _cswap(control: cirq.Qid, a: cirq.Qid, b: cirq.Qid, aux: cirq.Qid) -> cirq.OP_TREE:
            """A CSWAP with 4T complexity and whose adjoint has 0T complexity.

                A controlled SWAP that swaps `a` and `b` based on `control`.
            It uses an extra qubit `aux` so that its adjoint would have
            a T complexity of zero.
            """
            yield cirq.CNOT(a, b)
            yield and_gate.And(adjoint=self.adjoint).on(control, b, aux)
            yield cirq.CNOT(aux, a)
            yield cirq.CNOT(a, b)

        def _decomposition():
            # computes the difference of x - y where
            #   x = 2*x_msb + x_lsb
            #   y = 2*y_msb + y_lsb
            # And stores the result in x_lsb and y_lsb such that
            #   sign(x - y) = sign(x_lsb - y_lsb)
            # This decomposition uses 3 ancilla qubits in order to have a
            # T complexity of 8.
            yield cirq.X(ancilla[0])
            yield cirq.CNOT(y_msb, x_msb)
            yield cirq.CNOT(y_lsb, x_lsb)
            yield from _cswap(x_msb, x_lsb, ancilla[0], ancilla[1])
            yield from _cswap(x_msb, y_msb, y_lsb, ancilla[2])
            yield cirq.CNOT(y_lsb, x_lsb)

        if self.adjoint:
            yield from reversed(tuple(cirq.flatten_to_ops(_decomposition())))
        else:
            yield from _decomposition()

    def __pow__(self, power: int) -> cirq.Gate:
        if power == 1:
            return self
        if power == -1:
            return BiQubitsMixer(adjoint=not self.adjoint)
        return NotImplemented  # pragma: no cover

    def _t_complexity_(self) -> infra.TComplexity:
        if self.adjoint:
            return infra.TComplexity(clifford=18)
        return infra.TComplexity(t=8, clifford=28)

    def _has_unitary_(self):
        return not self.adjoint


@attr.frozen
class SingleQubitCompare(infra.GateWithRegisters):
    """Applies U|a>|b>|0>|0> = |a> |a=b> |(a<b)> |(a>b)>

    Source: (FIG. 3) in https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
    """  # pylint: disable=line-too-long

    adjoint: bool = False

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(a=1, b=1, less_than=1, greater_than=1)

    def __repr__(self) -> str:
        return f'cirq_ft.algos.SingleQubitCompare({self.adjoint})'

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        a = quregs['a']
        b = quregs['b']
        less_than = quregs['less_than']
        greater_than = quregs['greater_than']

        def _decomposition() -> Iterator[cirq.Operation]:
            yield and_gate.And((0, 1), adjoint=self.adjoint).on(*a, *b, *less_than)
            yield cirq.CNOT(*less_than, *greater_than)
            yield cirq.CNOT(*b, *greater_than)
            yield cirq.CNOT(*a, *b)
            yield cirq.CNOT(*a, *greater_than)
            yield cirq.X(*b)

        if self.adjoint:
            yield from reversed(tuple(_decomposition()))
        else:
            yield from _decomposition()

    def __pow__(self, power: int) -> cirq.Gate:
        if not isinstance(power, int):
            raise ValueError('SingleQubitCompare is only defined for integer powers.')
        if power % 2 == 0:
            return cirq.IdentityGate(4)
        if power < 0:
            return SingleQubitCompare(adjoint=not self.adjoint)
        return self

    def _t_complexity_(self) -> infra.TComplexity:
        if self.adjoint:
            return infra.TComplexity(clifford=11)
        return infra.TComplexity(t=4, clifford=16)


def _equality_with_zero(
    context: cirq.DecompositionContext, qubits: Sequence[cirq.Qid], z: cirq.Qid
) -> cirq.OP_TREE:
    if len(qubits) == 1:
        (q,) = qubits
        yield cirq.X(q)
        yield cirq.CNOT(q, z)
        return

    ancilla = context.qubit_manager.qalloc(len(qubits) - 2)
    yield and_gate.And(cv=[0] * len(qubits)).on(*qubits, *ancilla, z)


@attr.frozen
class LessThanEqualGate(cirq.ArithmeticGate):
    """Applies U|x>|y>|z> = |x>|y> |z ^ (x <= y)>"""

    x_bitsize: int
    y_bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.x_bitsize, [2] * self.y_bitsize, [2]

    def with_registers(self, *new_registers) -> "LessThanEqualGate":
        return LessThanEqualGate(len(new_registers[0]), len(new_registers[1]))

    def apply(self, *register_vals: int) -> Union[int, int, Iterable[int]]:
        x_val, y_val, target_val = register_vals
        return x_val, y_val, target_val ^ (x_val <= y_val)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.x_bitsize
        wire_symbols += ["In(y)"] * self.y_bitsize
        wire_symbols += ['+(x <= y)']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover

    def __repr__(self) -> str:
        return f'cirq_ft.LessThanEqualGate({self.x_bitsize}, {self.y_bitsize})'

    def _decompose_via_tree(
        self, context: cirq.DecompositionContext, X: Sequence[cirq.Qid], Y: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Returns comparison oracle from https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf

        This decomposition follows the tree structure of (FIG. 2)
        """  # pylint: disable=line-too-long
        if len(X) == 1:
            return
        if len(X) == 2:
            yield BiQubitsMixer().on_registers(x=X, y=Y, ancilla=context.qubit_manager.qalloc(3))
            return

        m = len(X) // 2
        yield self._decompose_via_tree(context, X[:m], Y[:m])
        yield self._decompose_via_tree(context, X[m:], Y[m:])
        yield BiQubitsMixer().on_registers(
            x=(X[m - 1], X[-1]), y=(Y[m - 1], Y[-1]), ancilla=context.qubit_manager.qalloc(3)
        )

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        """Decomposes the gate in a T-complexity optimal way.

        The construction can be broken in 4 parts:
            1. In case of differing bitsizes then a multicontrol And Gate
                - Section III.A. https://arxiv.org/abs/1805.03662) is used to check whether
                the extra prefix is equal to zero:
                    - result stored in: `prefix_equality` qubit.
            2. The tree structure (FIG. 2) https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
                followed by a SingleQubitCompare to compute the result of comparison of
                the suffixes of equal length:
                    - result stored in: `less_than` and `greater_than` with equality in qubits[-2]
            3. The results from the previous two steps are combined to update the target qubit.
            4. The adjoint of the previous operations is added to restore the input qubits
                to their original state and clean the ancilla qubits.
        """  # pylint: disable=line-too-long

        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())

        lhs, rhs, target = qubits[: self.x_bitsize], qubits[self.x_bitsize : -1], qubits[-1]

        n = min(len(lhs), len(rhs))

        prefix_equality = None
        adjoint: List[cirq.Operation] = []

        # if one of the registers is longer than the other store equality with |0--0>
        # into `prefix_equality` using d = |len(P) - len(Q)| And operations => 4d T.
        if len(lhs) != len(rhs):
            (prefix_equality,) = context.qubit_manager.qalloc(1)
            if len(lhs) > len(rhs):
                for op in cirq.flatten_to_ops(
                    _equality_with_zero(context, lhs[:-n], prefix_equality)
                ):
                    yield op
                    adjoint.append(cirq.inverse(op))
            else:
                for op in cirq.flatten_to_ops(
                    _equality_with_zero(context, rhs[:-n], prefix_equality)
                ):
                    yield op
                    adjoint.append(cirq.inverse(op))

                yield cirq.X(target), cirq.CNOT(prefix_equality, target)

        # compare the remaing suffix of P and Q
        lhs = lhs[-n:]
        rhs = rhs[-n:]
        for op in cirq.flatten_to_ops(self._decompose_via_tree(context, lhs, rhs)):
            yield op
            adjoint.append(cirq.inverse(op))

        less_than, greater_than = context.qubit_manager.qalloc(2)
        yield SingleQubitCompare().on_registers(
            a=lhs[-1], b=rhs[-1], less_than=less_than, greater_than=greater_than
        )
        adjoint.append(
            SingleQubitCompare(adjoint=True).on_registers(
                a=lhs[-1], b=rhs[-1], less_than=less_than, greater_than=greater_than
            )
        )

        if prefix_equality is None:
            yield cirq.X(target)
            yield cirq.CNOT(greater_than, target)
        else:
            (less_than_or_equal,) = context.qubit_manager.qalloc(1)
            yield and_gate.And([1, 0]).on(prefix_equality, greater_than, less_than_or_equal)
            adjoint.append(
                and_gate.And([1, 0], adjoint=True).on(
                    prefix_equality, greater_than, less_than_or_equal
                )
            )

            yield cirq.CNOT(less_than_or_equal, target)

        yield from reversed(adjoint)

    def _t_complexity_(self) -> infra.TComplexity:
        n = min(self.x_bitsize, self.y_bitsize)
        d = max(self.x_bitsize, self.y_bitsize) - n
        is_second_longer = self.y_bitsize > self.x_bitsize
        if d == 0:
            # When both registers are of the same size the T complexity is
            # 8n - 4 same as in https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf.  pylint: disable=line-too-long
            return infra.TComplexity(t=8 * n - 4, clifford=46 * n - 17)
        else:
            # When the registers differ in size and `n` is the size of the smaller one and
            # `d` is the difference in size. The T complexity is the sum of the tree
            # decomposition as before giving 8n + O(1) and the T complexity of an `And` gate
            # over `d` registers giving 4d + O(1) totaling 8n + 4d + O(1).
            # From the decomposition we get that the constant is -4 as well as the clifford counts.
            if d == 1:
                return infra.TComplexity(t=8 * n, clifford=46 * n + 3 + 2 * is_second_longer)
            else:
                return infra.TComplexity(
                    t=8 * n + 4 * d - 4, clifford=46 * n + 17 * d - 14 + 2 * is_second_longer
                )

    def _has_unitary_(self):
        return True


@attr.frozen
class ContiguousRegisterGate(cirq.ArithmeticGate):
    """Applies U|x>|y>|0> -> |x>|y>|x(x-1)/2 + y>

    This is useful in the case when $|x>$ and $|y>$ represent two selection registers such that
     $y < x$. For example, imagine a classical for-loop over two variables $x$ and $y$:

    >>> N = 10
    >>> data = [[1000 * x +  10 * y for y in range(x)] for x in range(N)]
    >>> for x in range(N):
    ...     for y in range(x):
    ...         # Iterates over a total of (N * (N - 1)) / 2 elements.
    ...         assert data[x][y] == 1000 * x + 10 * y

    We can rewrite the above using a single for-loop that uses a "contiguous" variable `i` s.t.

    >>> import numpy as np
    >>> N = 10
    >>> data = [[1000 * x + 10 * y for y in range(x)] for x in range(N)]
    >>> for i in range((N * (N - 1)) // 2):
    ...     x = int(np.floor((1 + np.sqrt(1 + 8 * i)) / 2))
    ...     y = i - (x * (x - 1)) // 2
    ...     assert data[x][y] == 1000 * x + 10 * y

     Note that both the for-loops iterate over the same ranges and in the same order. The only
     difference is that the second loop is a "flattened" version of the first one.

     Such a flattening of selection registers is useful when we want to load multi dimensional
     data to a target register which is indexed on selection registers $x$ and $y$ such that
     $0<= y <= x < N$ and we want to use a `SelectSwapQROM` to laod this data; which gives a
     sqrt-speedup over a traditional QROM at the cost of using more memory and loading chunks
     of size `sqrt(N)` in a single iteration. See the reference for more details.

     References:
         [Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction]
         (https://arxiv.org/abs/2011.03494)
            Lee et. al. (2020). Appendix F, Page 67.
    """

    bitsize: int
    target_bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize, [2] * self.target_bitsize

    def with_registers(self, *new_registers) -> 'ContiguousRegisterGate':
        x_bitsize, y_bitsize, target_bitsize = [len(reg) for reg in new_registers]
        assert (
            x_bitsize == y_bitsize
        ), f'x_bitsize={x_bitsize} should be same as y_bitsize={y_bitsize}'
        return ContiguousRegisterGate(x_bitsize, target_bitsize)

    def apply(self, *register_vals: int) -> Union[int, Iterable[int]]:
        x, y, target = register_vals
        return x, y, target ^ ((x * (x - 1)) // 2 + y)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += ["In(y)"] * self.bitsize
        wire_symbols += ['+(x(x-1)/2 + y)'] * self.target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> infra.TComplexity:
        # See the linked reference for explanation of the Toffoli complexity.
        toffoli_complexity = infra.t_complexity(cirq.CCNOT)
        return (self.bitsize**2 + self.bitsize - 1) * toffoli_complexity

    def __repr__(self) -> str:
        return f'cirq_ft.ContiguousRegisterGate({self.bitsize}, {self.target_bitsize})'

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover


@attr.frozen
class AdditionGate(cirq.ArithmeticGate):
    """Applies U|p>|q> -> |p>|p+q>.

    Args:
        bitsize: The number of bits used to represent each integer p and q.
            Note that this adder does not detect overflow if bitsize is not
            large enough to hold p + q and simply drops the most significant bits.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize

    def with_registers(self, *new_registers) -> 'AdditionGate':
        return AdditionGate(len(new_registers[0]))

    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        p, q = register_values
        return p, p + q

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += ["In(y)/Out(x+y)"] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self):
        return True

    def _left_building_block(self, inp, out, anc, depth):
        if depth == self.bitsize - 1:
            return
        else:
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(anc[depth - 1], out[depth])
            yield and_gate.And().on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield from self._left_building_block(inp, out, anc, depth + 1)

    def _right_building_block(self, inp, out, anc, depth):
        if depth == 0:
            return
        else:
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield and_gate.And(adjoint=True).on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(inp[depth], out[depth])
            yield from self._right_building_block(inp, out, anc, depth - 1)

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        input_bits = qubits[: self.bitsize]
        output_bits = qubits[self.bitsize :]
        ancillas = context.qubit_manager.qalloc(self.bitsize - 1)
        # Start off the addition by anding into the ancilla
        yield and_gate.And().on(input_bits[0], output_bits[0], ancillas[0])
        # Left part of Fig.2
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        yield cirq.CX(ancillas[-1], output_bits[-1])
        yield cirq.CX(input_bits[-1], output_bits[-1])
        # right part of Fig.2
        yield from self._right_building_block(input_bits, output_bits, ancillas, self.bitsize - 2)
        yield and_gate.And(adjoint=True).on(input_bits[0], output_bits[0], ancillas[0])
        yield cirq.CX(input_bits[0], output_bits[0])
        context.qubit_manager.qfree(ancillas)

    def _t_complexity_(self) -> infra.TComplexity:
        # There are N - 2 building blocks each with one And/And^dag contributing 13 cliffords and
        # 6 CXs. In addition there is one additional And/And^dag pair and 3 CXs.
        num_clifford = (self.bitsize - 2) * 19 + 16
        return infra.TComplexity(t=4 * self.bitsize - 4, clifford=num_clifford)

    def __repr__(self) -> str:
        return f'cirq_ft.AdditionGate({self.bitsize})'


@attr.frozen(auto_attribs=True)
class AddMod(cirq.ArithmeticGate):
    """Applies U_{M}_{add}|x> = |(x + add) % M> if x < M else |x>.

    Applies modular addition to input register `|x>` given parameters `mod` and `add_val` s.t.
        1) If integer `x` < `mod`: output is `|(x + add) % M>`
        2) If integer `x` >= `mod`: output is `|x>`.

    This condition is needed to ensure that the mapping of all input basis states (i.e. input
    states |0>, |1>, ..., |2 ** bitsize - 1) to corresponding output states is bijective and thus
    the gate is reversible.

    Also supports controlled version of the gate by specifying a per qubit control value as a tuple
    of integers passed as `cv`.
    """

    bitsize: int
    mod: int = attr.field()
    add_val: int = 1
    cv: Tuple[int, ...] = attr.field(converter=infra.to_tuple, default=())

    @mod.validator
    def _validate_mod(self, attribute, value):
        if not 1 <= value <= 2**self.bitsize:
            raise ValueError(f"mod: {value} must be between [1, {2 ** self.bitsize}].")

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        add_reg = (2,) * self.bitsize
        control_reg = (2,) * len(self.cv)
        return (control_reg, add_reg) if control_reg else (add_reg,)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> "AddMod":
        raise NotImplementedError()

    def apply(self, *args) -> Union[int, Iterable[int]]:
        target_val = args[-1]
        if target_val < self.mod:
            new_target_val = (target_val + self.add_val) % self.mod
        else:
            new_target_val = target_val
        if self.cv and args[0] != int(''.join(str(x) for x in self.cv), 2):
            new_target_val = target_val
        ret = (args[0], new_target_val) if self.cv else (new_target_val,)
        return ret

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if b else '@(0)' for b in self.cv]
        wire_symbols += [f"Add_{self.add_val}_Mod_{self.mod}"] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int) -> 'AddMod':
        return AddMod(self.bitsize, self.mod, add_val=self.add_val * power, cv=self.cv)

    def __repr__(self) -> str:
        return f'cirq_ft.AddMod({self.bitsize}, {self.mod}, {self.add_val}, {self.cv})'

    def _t_complexity_(self) -> infra.TComplexity:
        # Rough cost as given in https://arxiv.org/abs/1905.09749
        return 5 * infra.t_complexity(AdditionGate(self.bitsize))

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

import abc
from typing import Dict, Iterator, List, Sequence, Tuple

import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate


def _unary_iteration_segtree(
    ops: List[cirq.Operation],
    control: cirq.Qid,
    selection: Sequence[cirq.Qid],
    sl: int,
    l: int,
    r: int,
    l_iter: int,
    r_iter: int,
    qm: cirq.QubitManager,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    if l >= r_iter or l_iter >= r:
        # Range corresponding to this node is completely outside of iteration range.
        return
    if l == (r - 1):
        # Reached a leaf node; yield the operations.
        yield tuple(ops), control, l
        ops.clear()
        return
    assert sl < len(selection)
    m = (l + r) >> 1
    if r_iter <= m:
        # Yield only left sub-tree.
        yield from _unary_iteration_segtree(
            ops, control, selection, sl + 1, l, m, l_iter, r_iter, qm
        )
        return
    if l_iter >= m:
        # Yield only right sub-tree
        yield from _unary_iteration_segtree(
            ops, control, selection, sl + 1, m, r, l_iter, r_iter, qm
        )
        return
    anc, sq = qm.qalloc(1)[0], selection[sl]
    ops.append(and_gate.And((1, 0)).on(control, sq, anc))
    yield from _unary_iteration_segtree(ops, anc, selection, sl + 1, l, m, l_iter, r_iter, qm)
    ops.append(cirq.CNOT(control, anc))
    yield from _unary_iteration_segtree(ops, anc, selection, sl + 1, m, r, l_iter, r_iter, qm)
    ops.append(and_gate.And(adjoint=True).on(control, sq, anc))
    qm.qfree([anc])


def _unary_iteration_zero_control(
    ops: List[cirq.Operation],
    selection: Sequence[cirq.Qid],
    l_iter: int,
    r_iter: int,
    qm: cirq.QubitManager,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    sl, l, r = 0, 0, 2 ** len(selection)
    m = (l + r) >> 1
    ops.append(cirq.X(selection[0]))
    yield from _unary_iteration_segtree(
        ops, selection[0], selection[1:], sl, l, m, l_iter, r_iter, qm
    )
    ops.append(cirq.X(selection[0]))
    yield from _unary_iteration_segtree(
        ops, selection[0], selection[1:], sl, m, r, l_iter, r_iter, qm
    )


def _unary_iteration_single_control(
    ops: List[cirq.Operation],
    control: cirq.Qid,
    selection: Sequence[cirq.Qid],
    l_iter: int,
    r_iter: int,
    qm: cirq.QubitManager,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    sl, l, r = 0, 0, 2 ** len(selection)
    yield from _unary_iteration_segtree(ops, control, selection, sl, l, r, l_iter, r_iter, qm)


def _unary_iteration_multi_controls(
    ops: List[cirq.Operation],
    controls: Sequence[cirq.Qid],
    selection: Sequence[cirq.Qid],
    l_iter: int,
    r_iter: int,
    qm: cirq.QubitManager,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    num_controls = len(controls)
    and_ancilla = qm.qalloc(num_controls - 2)
    and_target = qm.qalloc(1)[0]
    multi_controlled_and = and_gate.And((1,) * len(controls)).on_registers(
        control=controls, ancilla=and_ancilla, target=and_target
    )
    ops.append(multi_controlled_and)
    yield from _unary_iteration_single_control(ops, and_target, selection, l_iter, r_iter, qm)
    ops.append(cirq.inverse(multi_controlled_and))
    qm.qfree(and_ancilla + [and_target])


def unary_iteration(
    l_iter: int,
    r_iter: int,
    flanking_ops: List[cirq.Operation],
    controls: Sequence[cirq.Qid],
    selection: Sequence[cirq.Qid],
    qubit_manager: cirq.QubitManager,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    """The method performs unary iteration on `selection` integer in `range(l_iter, r_iter)`.

    Unary iteration is a coherent for loop that can be used to conditionally perform a different
    operation on a target register for every integer in the `range(l_iter, r_iter)` stored in the
    selection register.

    Users can write multi-dimensional coherent for loops as follows:

    >>> N, M = 5, 7
    >>> target = [[cirq.q(f't({i}, {j})') for j in range(M)] for i in range(N)]
    >>> selection = [[cirq.q(f's({i}, {j})') for j in range(3)] for i in range(3)]
    >>> circuit = cirq.Circuit()
    >>> i_ops = []
    >>> for i_optree, i_control, i in unary_iteration(0, N, i_ops, [], selection[0]):
    >>>     circuit.append(i_optree)
    >>>     j_ops = []
    >>>     for j_optree, j_control, j in unary_iteration(0, M, j_ops, [i_control], selection[1]):
    >>>         circuit.append(j_optree)
    >>>         # Conditionally perform operations on target register using `j_control`, `i` & `j`.
    >>>         circuit.append(cirq.CNOT(j_control, target[i][j]))
    >>>     circuit.append(j_ops)
    >>> circuit.append(i_ops)

    Args:
        l_iter: Starting index of the iteration range.
        r_iter: Ending index of the iteration range.
        flanking_ops: A list of `cirq.Operation`s that represents operations to be inserted in the
            circuit before/after the first/last iteration of the unary iteration for loop. Note that
            the list is mutated by the function, such that before calling the function, the list
            represents operations to be inserted before the first iteration and after the last call
            to the function, list represents operations to be inserted at the end of last iteration.
        controls: Control register of qubits.
        selection: Selection register of qubits.
        qubit_manager: A `cirq.QubitManager` to allocate new qubits.

    Yields:
        (r_iter - l_iter) different tuples, each corresponding to an integer in range
        [l_iter, r_iter).
        Each returned tuple also corresponds to a unique leaf in the unary iteration tree.
        The values of yielded `Tuple[cirq.OP_TREE, cirq.Qid, int]` correspond to:
        - cirq.OP_TREE: The op-tree to be inserted in the circuit to get to the current leaf.
        - cirq.Qid: Control qubit used to conditionally apply operations on the target conditioned
            on the returned integer.
        - int: The current integer in the iteration `range(l_iter, r_iter)`.
    """
    assert 2 ** len(selection) >= r_iter - l_iter
    assert len(selection) > 0
    if len(controls) == 0:
        yield from _unary_iteration_zero_control(
            flanking_ops, selection, l_iter, r_iter, qubit_manager
        )
    elif len(controls) == 1:
        yield from _unary_iteration_single_control(
            flanking_ops, controls[0], selection, l_iter, r_iter, qubit_manager
        )
    else:
        yield from _unary_iteration_multi_controls(
            flanking_ops, controls, selection, l_iter, r_iter, qubit_manager
        )


class UnaryIterationGate(infra.GateWithRegisters):
    @cached_property
    @abc.abstractmethod
    def control_registers(self) -> infra.Registers:
        pass

    @cached_property
    @abc.abstractmethod
    def selection_registers(self) -> infra.SelectionRegisters:
        pass

    @cached_property
    @abc.abstractmethod
    def target_registers(self) -> infra.Registers:
        pass

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    @cached_property
    def extra_registers(self) -> infra.Registers:
        return infra.Registers([])

    @abc.abstractmethod
    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        """Apply nth operation on the target registers when selection registers store `n`.

        The `UnaryIterationGate` class is a mixin that represents a coherent for-loop over
        different indices (i.e. selection registers). This method denotes the "body" of the
        for-loop, which is executed `self.selection_registers.total_iteration_size` times and each
        iteration represents a unique combination of values stored in selection registers. For each
        call, the method should return the operations that should be applied to the target
        registers, given the values stored in selection registers.

        The derived classes should specify the following arguments as `**kwargs`:
            1) `control: cirq.Qid`: A qubit which can be used as a control to selectively
            apply operations when selection register stores specific value.
            2) Register names in `self.selection_registers`: Each argument corresponds to
            a selection register and represents the integer value stored in the register.
            3) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            4) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """

    def decompose_zero_selection(
        self, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Specify decomposition of the gate when selection register is empty

        By default, if the selection register is empty, the decomposition will raise a
        `NotImplementedError`. The derived classes can override this method and specify
        a custom decomposition that should be used if the selection register is empty,
        i.e. `self.selection_registers.bitsize == 0`.

        The derived classes should specify the following arguments as `**kwargs`:
            1) Register names in `self.control_registers`: Each argument corresponds to a
            control register and represents sequence of qubits that represent the control register.
            2) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            3) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """
        raise NotImplementedError("Selection register must not be empty.")

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        if self.selection_registers.bitsize == 0:
            return self.decompose_zero_selection(context=context, **quregs)

        num_loops = len(self.selection_registers)
        target_regs = {k: v for k, v in quregs.items() if k in self.target_registers}
        extra_regs = {k: v for k, v in quregs.items() if k in self.extra_registers}

        def unary_iteration_loops(
            nested_depth: int,
            selection_reg_name_to_val: Dict[str, int],
            controls: Sequence[cirq.Qid],
        ) -> Iterator[cirq.OP_TREE]:
            """Recursively write any number of nested coherent for-loops using unary iteration.

            This helper method is useful to write `num_loops` number of nested coherent for-loops by
            recursively calling this method `num_loops` times. The ith recursive call of this method
            has `nested_depth=i` and represents the body of ith nested for-loop.

            Args:
                nested_depth: Integer between `[0, num_loops]` representing the nest-level of
                    for-loop for which this method implements the body.
                selection_reg_name_to_val: A dictionary containing `nested_depth` elements mapping
                    the selection integer names (i.e. loop variables) to corresponding values;
                    for each of the `nested_depth` parent for-loops written before.
                controls: Control qubits that should be used to conditionally activate the body of
                    this for-loop.

            Returns:
                `cirq.OP_TREE` implementing `num_loops` nested coherent for-loops, with operations
                returned by `self.nth_operation` applied conditionally to the target register based
                on values of selection registers.
            """
            if nested_depth == num_loops:
                yield self.nth_operation(
                    context=context,
                    control=controls[0],
                    **selection_reg_name_to_val,
                    **target_regs,
                    **extra_regs,
                )
                return
            # Use recursion to write `num_loops` nested loops using unary_iteration().
            ops: List[cirq.Operation] = []
            ith_for_loop = unary_iteration(
                l_iter=0,
                r_iter=self.selection_registers[nested_depth].iteration_length,
                flanking_ops=ops,
                controls=controls,
                selection=quregs[self.selection_registers[nested_depth].name],
                qubit_manager=context.qubit_manager,
            )
            for op_tree, control_qid, n in ith_for_loop:
                yield op_tree
                selection_reg_name_to_val[self.selection_registers[nested_depth].name] = n
                yield from unary_iteration_loops(
                    nested_depth + 1, selection_reg_name_to_val, (control_qid,)
                )
            yield ops

        return unary_iteration_loops(0, {}, self.control_registers.merge_qubits(**quregs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Basic circuit diagram.

        Descendants are encouraged to override this with more descriptive
        circuit diagram information.
        """
        wire_symbols = ["@"] * self.control_registers.bitsize
        wire_symbols += ["In"] * self.selection_registers.bitsize
        wire_symbols += [self.__class__.__name__] * self.target_registers.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
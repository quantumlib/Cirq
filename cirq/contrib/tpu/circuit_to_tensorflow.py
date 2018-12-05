# Copyright 2018 Google LLC
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

from typing import Any, Callable, Dict, List, NamedTuple, Tuple, TypeVar, Union

from collections import namedtuple

import numpy as np
import tensorflow as tf

from cirq import ops, circuits, linalg, protocols, optimizers


# We logically divide the qubits into groups of this size, and operate on each
# group as a multi-qubit register. This significantly improves compilation
# times, and avoids padding overhead when the tensors are laid out into memory.
# This is an internal detail not exposed to callers (e.g. we reshape before
# returning).
_GROUP_SIZE = 7


ComputeFuncAndFeedDict = NamedTuple(
    'ComputeFuncAndFeedDict',
    [
        ('compute', Callable[[], tf.Tensor]),
        ('feed_dict', Dict[tf.Tensor, Any])
    ])


def circuit_to_tensorflow_runnable(
        circuit: circuits.Circuit,
        initial_state: Union[int, np.ndarray] = 0,
        ) -> ComputeFuncAndFeedDict:
    """Returns a compute function and feed_dict for a `cirq.Circuit`'s output.

    `result.compute()` will return a `tensorflow.Tensor` with
    `tensorflow.placeholder` objects to be filled in by `result.feed_dict`, at
    which point it will evaluate to the output state vector of the circuit.

    You can apply further operations to the tensor returned by
    `result.compute`. This allows, for example, for the final result to be
    a small amount of computed data (e.g. an expectation value) instead of the
    gigantic raw state vector.

    The tensor returned by `result.compute` is intended to run efficiently
    on cloud TPUs. It will have dtype complex64 and a shape of (2**n,) where n
    is the number of qubits.

    Examples:
        To simulate the circuit with tensorflow in a normal session, forward
        this method's output into `tensorflow.Session.run` as follows:

            import tensorflow as tf
            r = circuit_to_tensorflow_runnable(...)
            with tf.Session() as session:
                output = session.run(r.compute(), feed_dict=r.feed_dict)
            print(output)

        Note that you can use the returned tensor in further computations. For
        example, to compute the chance of the system ending up in the first 128
        computational basis states you can use `tf.norm(tensor[:128], 2)`:

            import tensorflow as tf
            r = circuit_to_tensorflow_runnable(...)
            expectation = lambda: tf.norm(r.compute()[:128], 2)
            with tf.Session() as session:
                output = session.run(expectation, feed_dict=r.feed_dict)
            print(output)

        For documentation on running against cloud TPUs, see

            https://cloud.google.com/tpu/docs/quickstart#run_example

        Generally speaking, from within a cloud instance, you use
        `tf.contrib.tpu.rewrite` to convert the tensor into a TPU compatible
        form, initialize the TPU system, then run the rewritten tensor:

            import tensorflow as tf
            TPU_TARGET = ???????
            r = circuit_to_tensorflow_runnable(...YOUR_CIRCUIT...)
            rewritten_for_tpu = tf.contrib.tpu.rewrite(r.compute)

            with tf.Session(target=TPU_TARGET) as session:
                session.run(tf.contrib.tpu.initialize_system())
                output = session.run(rewritten_for_tpu, feed_dict=r.feed_dict)
            print(output)

    Args:
        circuit: The circuit to apply to `initial_state` to produce an output
            state vector.
        initial_state: The input into the circuit. If this is an integer, it
            indicates that the input state is a computational basis state where
            the k'th qubit is set by the k'th bit of the integer. If this is
            a numpy array, it should directly encode a normalized wavefunction.

    Returns:
        A ComputeFuncAndFeedDict, which is a named tuple whose first element is
        a function that returns a Tensor representing the output state vector
        that results from applying the given circuit to the given, and whose
        second element is a feed_dict containing important parameters describing
        that tensor.
    """
    if not circuit.are_all_measurements_terminal():
        raise ValueError('not circuit.are_all_measurements_terminal()')

    t = _TensorCircuit(circuit, initial_state)
    return ComputeFuncAndFeedDict(t.compute, t.feed_dict)


_TransformsThenCzs = namedtuple(
    '_GroupTransformThenInteract',
    ['group_matrices', 'cz_indices'])


class _QubitGrouping:
    """Decides how to combine qubits into groups to maximize performance."""

    def __init__(self, circuit: circuits.Circuit) -> None:
        self.qubits = list(ops.QubitOrder.DEFAULT.order_for(
            circuit.all_qubits()))

        # Pick group sizes.
        d = len(self.qubits) // _GROUP_SIZE
        r = len(self.qubits) % _GROUP_SIZE

        # Prefer to put the odd-sized group in the middle, or at the start.
        # (Because compilers may pad the last group up to a fixed size.)
        sizes = [_GROUP_SIZE] * d
        if r:
            sizes.append(r)
            sizes[-2:] = reversed(sizes[-2:])

        # Make the actual groups.
        self.groups = []  # type: List[List[ops.QubitId]]
        i = 0
        for s in sizes:
            self.groups.append(self.qubits[i:i+s])
            i += s

        # Make it easy to lookup where a  qubit goes.
        self.map = {qubit: (group_id, item_id)
                    for group_id, group in enumerate(self.groups)
                    for item_id, qubit in enumerate(group)}

    def loc(self, q: ops.QubitId) -> Tuple[int, int]:
        return self.map[q]

    def ind(self, q: ops.QubitId) -> int:
        g, a = self.loc(q)
        past = sum(len(h) for h in self.groups[:g])
        return self.qubit_count() - 1 - a - past

    def qubit_count(self) -> int:
        return sum(len(g) for g in self.groups)

    def system_size(self) -> int:
        return 1 << self.qubit_count()

    def flat_shape(self) -> Tuple[int]:
        return self.system_size(),

    def all_in_same_group(self, *qubits: ops.QubitId) -> bool:
        return len({self.loc(q)[0] for q in qubits}) <= 1

    def decompose_keep_func(self, op: ops.Operation) -> bool:
        # Keep CZs.
        if len(op.qubits) == 2 and op == ops.CZ(*op.qubits):
            return True

        # Keep within-group operation with a known unitary
        if self.all_in_same_group(*op.qubits):
            if protocols.unitary(op, None) is not None:
                return True

        return False

    def intercept_decompose_func(self,
                                 op: Union[ops.Operation, circuits.Circuit]
                                 ) -> ops.OP_TREE:
        if not isinstance(op, ops.Operation):
            return NotImplemented

        # Drop measurements.
        if ops.MeasurementGate.is_measurement(op):
            return []

        # Immediately completely decompose two qubit unitary operations.
        if len(op.qubits) == 2:
            m = protocols.unitary(op, None)
            if m is not None:
                return optimizers.two_qubit_matrix_to_operations(
                    op.qubits[0],
                    op.qubits[1],
                    mat=m,
                    allow_partial_czs=False
                )

        # Fallback to default.
        return NotImplemented


def _circuit_as_layers(circuit: circuits.Circuit,
                       grouping: _QubitGrouping) -> List[_TransformsThenCzs]:
    """Transforms a circuit into a series of GroupMatrix+CZ layers.

    Args:
        circuit: The circuit to transform.
        grouping: How the circuit's qubits are combined into groups.

    Returns:
        A list of layers. Each layer has a matrix to apply to each group of
        qubits, and a list of CZs to apply to pairs of qubits crossing
        between groups.
    """
    frontier = {q: 0 for q in circuit.all_qubits()}

    layers = []
    while True:
        # Pull within-group operations into per-group matrices.
        any_group_matrices = False
        group_matrices = []
        for g in grouping.groups:
            # Scan for reachable operations contained within the group qubits.
            start_frontier = {q: frontier[q] for q in g}
            end_frontier = circuit.reachable_frontier_from(start_frontier)
            mergeable_ops = circuit.findall_operations_between(start_frontier,
                                                               end_frontier)

            # Advance frontier.
            for q, v in end_frontier.items():
                frontier[q] = v

            # Fold reachable operations into a single group matrix.
            group_matrix = np.eye(1 << len(g)).reshape((2, 2) * len(g))
            if mergeable_ops:
                any_group_matrices = True
            for _, op in mergeable_ops:
                group_matrix = linalg.targeted_left_multiply(
                    left_matrix=protocols.unitary(op).reshape(
                        (2, 2) * len(op.qubits)),
                    right_target=group_matrix,
                    target_axes=[grouping.loc(q)[1] for q in op.qubits])
            group_matrices.append(np.transpose(group_matrix.reshape(
                1 << len(g), 1 << len(g))))

        # Scan for reachable CZ operations between groups.
        end_frontier = circuit.reachable_frontier_from(
            frontier,
            is_blocker=lambda op: grouping.all_in_same_group(*op.qubits))
        cz_ops = circuit.findall_operations_between(frontier, end_frontier)

        # Advance frontier.
        frontier = end_frontier

        # List out qubit index pairs for each CZ.
        cz_indices = []
        for _, cz in cz_ops:
            a, b = cz.qubits
            assert cz == ops.CZ(a, b)
            cz_indices.append((grouping.ind(a), grouping.ind(b)))

        # Combine group and CZ operations into a simulation layer.
        if not any_group_matrices and not cz_indices:
            break
        layer = _TransformsThenCzs(group_matrices=group_matrices,
                                   cz_indices=cz_indices)
        layers.append(layer)

    # We should have processed the whole circuit.
    assert frontier == {q: len(circuit) for q in circuit.all_qubits()}

    return layers


def _deref(tensor: tf.Tensor, index: tf.Tensor) -> tf.Tensor:
    """Equivalent to `tensor[index, ...]`.

    This is a workaround for XLA requiring constant tensor indices. It works
    by producing a node representing hardcoded instructions like the following:

       if index == 0: return tensor[0]
       if index == 1: return tensor[1]
       if index == 2: return tensor[2]
       .
       .
       if index == n-1: return tensor[n-1]

    This is acceptable as long as n*size(tensor) is negligible compared to
    the rest of the computation.
    """
    assert tensor.shape[0] > 0
    return _deref_helper(lambda i: tensor[i, ...],
                         index,
                         0,
                         tensor.shape[0] - 1)


def _multi_deref(tensors: List[tf.Tensor], index: tf.Tensor) -> List[tf.Tensor]:
    """Equivalent to `[t[index, ...] for t in tensors]`.

    See `_deref` for more details.
    """
    assert tensors
    assert tensors[0].shape[0] > 0
    return _deref_helper(lambda i: [tensor[i, ...] for tensor in tensors],
                         index,
                         0,
                         tensors[0].shape[0] - 1)


TItem = TypeVar('TItem')


def _deref_helper(func: Callable[[int], TItem],
                  index: tf.Tensor,
                  min_index: int,
                  max_index: int) -> TItem:
    assert min_index <= max_index
    if min_index == max_index:
        return func(min_index)
    mid = (min_index + max_index + 1) // 2
    return tf.cond(
        tf.math.greater_equal(index, mid),
        true_fn=lambda: _deref_helper(func, index, mid, max_index),
        false_fn=lambda: _deref_helper(func, index, min_index, mid - 1),
        strict=True)


class _TensorCircuit:
    def __init__(self,
                 circuit: circuits.Circuit,
                 initial_state: Union[int, np.ndarray]):
        self.grouping = _QubitGrouping(circuit)
        self.circuit = circuits.Circuit.from_ops(
            protocols.decompose(
                circuit,
                intercepting_decomposer=self.grouping.intercept_decompose_func,
                keep=self.grouping.decompose_keep_func))
        self.layers = _circuit_as_layers(self.circuit, self.grouping)
        self.feed_dict = {}  # type: Dict[tf.Tensor, Any]

        # Capture initial state.
        self._initial_state_func = self._pick_initial_state_func(initial_state)

        # Store list of CZ indices for each layer.
        max_czs = 0
        if self.layers:
            max_czs = max(len(e.cz_indices) for e in self.layers)
        max_czs = max(1, max_czs)
        self.secret_cz_indices = tf.placeholder(
            name='cz_indices',
            dtype=tf.int32,
            shape=(len(self.layers), max_czs, 2))
        padded_cz_indices = np.array(
            [
                e.cz_indices + [(-1, -1)] * (max_czs - len(e.cz_indices))
                for e in self.layers
            ],
            dtype=np.int32
        ).reshape((len(self.layers), max_czs, 2))
        self.feed_dict[self.secret_cz_indices] = padded_cz_indices

        # Store matrices for each qubit group at each layer.
        self.secret_group_matrices = [
            tf.placeholder(
                name='group_matrices_{}'.format(i),
                dtype=tf.complex64,
                shape=(len(self.layers),
                       1 << len(g),
                       1 << len(g)))
            for i, g in enumerate(self.grouping.groups)
        ]
        for i in range(len(self.grouping.groups)):
            j = len(self.grouping.groups) - i - 1
            self.feed_dict[self.secret_group_matrices[j]] = np.array([
                e.group_matrices[j]
                for e in self.layers
            ])

    def compute(self):
        v = self._initial_state_func()

        # Apply all layers.
        _, v = tf.while_loop(
            lambda i, w: i < len(self.layers),
            lambda i, w: (i + 1, self._after_layer(w, i)),
            [0, v])

        return tf.reshape(v, self.grouping.flat_shape())

    def _pick_initial_state_func(
            self,
            state: Union[int, np.ndarray]
            ) -> Callable[[], tf.Tensor]:
        if isinstance(state, int):
            secret_index = tf.placeholder(
                name='initial_computational_basis_state',
                dtype=tf.int32,
                shape=())
            self.feed_dict[secret_index] = state
            return lambda: tf.one_hot(secret_index,
                                      self.grouping.system_size(),
                                      dtype=tf.complex64)

        if isinstance(state, np.ndarray):
            secret_vector = tf.placeholder(name='initial_state_vector',
                                           dtype=tf.complex64,
                                           shape=state.shape)
            self.feed_dict[secret_vector] = state
            return lambda: secret_vector

        raise ValueError('Unsupported state type: {}'.format(type(state)))

    def _after_group_matrices(self,
                              v: tf.Tensor,
                              group_matrices: List[tf.Tensor]
                              ) -> tf.Tensor:
        s = [1 << len(g) for g in self.grouping.groups]
        t = self.grouping.system_size()
        for i in range(len(group_matrices)):
            inner_size = s[i]
            outer_size = t // inner_size
            v = tf.reshape(v, [inner_size, outer_size])
            v = tf.transpose(v)
            v = tf.matmul(v, group_matrices[i])
        return v

    def _after_czs(self, v: tf.Tensor, pairs: tf.Tensor) -> tf.Tensor:
        iota = tf.range(self.grouping.system_size())
        t = tf.constant(0, dtype=tf.int32)
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]
            index_mask = tf.bitwise.bitwise_or(
                tf.bitwise.left_shift(1, i),
                tf.bitwise.left_shift(1, j))
            index_mask = tf.cond(tf.math.equal(i, -1),
                                 lambda: -1,
                                 lambda: index_mask)
            masked_iota = tf.bitwise.bitwise_and(iota, index_mask)
            kept_iota = tf.math.equal(index_mask, masked_iota)
            t = tf.bitwise.bitwise_xor(t, tf.to_int32(kept_iota))
        negations = 1 - tf.to_complex64(t) * 2
        v *= negations
        return v

    def _after_layer(self, v: tf.Tensor, layer_index: tf.Tensor) -> tf.Tensor:

        if self.secret_group_matrices:
            group_matrices = _multi_deref(self.secret_group_matrices,
                                          layer_index)
            v = self._after_group_matrices(v, group_matrices)

        v = tf.reshape(v, self.grouping.flat_shape())
        if self.secret_cz_indices.shape[0] > 0:
            cz_indices = _deref(self.secret_cz_indices, layer_index)
            v = self._after_czs(v, cz_indices)

        return v

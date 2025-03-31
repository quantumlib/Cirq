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

import itertools
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import sympy

from cirq import circuits, linalg, ops, protocols, qis
from cirq.testing import lin_alg_utils


def highlight_text_differences(actual: str, expected: str) -> str:
    diff = ""
    for actual_line, desired_line in itertools.zip_longest(
        actual.splitlines(), expected.splitlines(), fillvalue=""
    ):
        diff += (
            "".join(
                a if a == b else "█"
                for a, b in itertools.zip_longest(actual_line, desired_line, fillvalue="")
            )
            + "\n"
        )
    return diff


def _measurement_subspaces(
    measured_qubits: Iterable[ops.Qid], n_qubits: int
) -> Sequence[Sequence[int]]:
    """Computes subspaces associated with projective measurement.

    The function computes a partitioning of the computational basis such
    that the subspace spanned by each partition corresponds to a distinct
    measurement outcome. In particular, if all qubits are measured then
    2**n singleton partitions are returned. If no qubits are measured then
    a single partition consisting of all basis states is returned.

    Args:
        measured_qubits: Qubits subject to measurement
        n_qubits: Total number of qubits in circuit
    Returns:
        Sequence of subspaces where each subspace is a sequence of
            computational basis states in order corresponding to qubit_order
    """

    # Consider projective measurement in the computational basis on a subset
    # of qubits. Each projection operator associated with the measurement is
    # uniquely determined by its range, here called a measurement subspace.
    #
    # Suppose that qubit q is not measured. Then computational basis states
    # whose indices have binary representations that differ only at position
    # q belong to the same measurement subspace. Generally, if computational
    # basis states a and b are such that
    #
    #     a & measurement_mask == b & measurement_mask
    #
    # then a and b belong to the same measurement subspace. In this case the
    # value of the expression on either side in the formula above is the
    # computational basis state in the measurement subspace containing
    # a and b which has the lowest index.
    measurement_mask = 0
    for i, _ in enumerate(sorted(measured_qubits)):
        measurement_mask |= 1 << i

    # Keyed by computational basis state with lowest index.
    measurement_subspaces: Dict[int, List[int]] = defaultdict(list)
    computational_basis = range(1 << n_qubits)

    for basis_state in computational_basis:
        subspace_key = basis_state & measurement_mask
        measurement_subspaces[subspace_key].append(basis_state)

    subspaces = list(measurement_subspaces.values())

    # Verify this is a partitioning (i.e. full coverage, no overlaps).
    assert sorted(itertools.chain(*subspaces)) == list(computational_basis)

    return subspaces


def assert_circuits_with_terminal_measurements_are_equivalent(
    actual: circuits.AbstractCircuit, reference: circuits.AbstractCircuit, atol: float = 1.0e-8
) -> None:
    """Determines if two circuits have equivalent effects.

    The circuits can contain measurements, but the measurements must be at the
    end of the circuit. Circuits are equivalent if, for all possible inputs,
    their outputs (classical bits for lines terminated with measurement and
    qubits for lines without measurement) are observationally indistinguishable
    up to a tolerance. Note that under this definition of equivalence circuits
    that differ solely in the overall phase of the post-measurement state of
    measured qubits are considered equivalent.

    For example, applying an extra Z gate to an unmeasured qubit changes the
    effect of a circuit. But inserting a Z gate operation just before a
    measurement does not.

    Args:
        actual: The circuit that was actually computed by some process.
        reference: A circuit with the correct function.
        atol: Absolute error tolerance.
    """
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    measured_qubits_actual = {
        qubit
        for op in actual.all_operations()
        if protocols.is_measurement(op)
        for qubit in op.qubits
    }
    measured_qubits_reference = {
        qubit
        for op in reference.all_operations()
        if protocols.is_measurement(op)
        for qubit in op.qubits
    }
    assert actual.are_all_measurements_terminal()
    assert reference.are_all_measurements_terminal()
    assert measured_qubits_actual == measured_qubits_reference

    all_qubits = actual.all_qubits().union(reference.all_qubits())

    matrix_actual = actual.unitary(qubits_that_should_be_present=all_qubits)
    matrix_reference = reference.unitary(qubits_that_should_be_present=all_qubits)

    n_qubits = len(all_qubits)
    n = matrix_actual.shape[0]
    assert n == 1 << n_qubits
    assert matrix_actual.shape == matrix_reference.shape == (n, n)

    # Consider the action of the two circuits Ca and Cr on state |x>:
    #
    #     |ya> = Ca|x>
    #     |yr> = Cr|x>
    #
    # Ca and Cr are equivalent according to the definition above iff
    # for each |x>:
    #  - probability of each measurement outcome is the same for |ya>
    #    and |yr> (across measured qubits),
    #  - amplitudes of each post-measurement state are the same for |ya>
    #    and |yr> except perhaps for an overall phase factor.
    #
    # These conditions are satisfied iff the matrices of the two circuits
    # are identical except perhaps for an overall phase factor for each
    # rectangular block spanning rows corresponding to the measurement
    # subspaces and all columns.
    #
    # Note two special cases of the rule above:
    #  - if no qubits are measured then the circuits are equivalent if
    #    their matrices are identical except for the global phase factor,
    #  - if all qubits are measured then the circuits are equivalent if
    #    their matrices differ by a diagonal unitary factor.
    subspaces = _measurement_subspaces(measured_qubits_actual, n_qubits)
    for subspace in subspaces:
        block_actual = matrix_actual[subspace, :]
        block_reference = matrix_reference[subspace, :]
        assert linalg.allclose_up_to_global_phase(block_actual, block_reference, atol=atol), (
            "Circuit's effect differs from the reference circuit.\n"
            '\n'
            'Diagram of actual circuit:\n'
            f'{actual}\n'
            '\n'
            'Diagram of reference circuit with desired function:\n'
            f'{reference}\n'
        )


def assert_same_circuits(
    actual: circuits.AbstractCircuit, expected: circuits.AbstractCircuit
) -> None:
    """Asserts that two circuits are identical, with a descriptive error.

    Args:
        actual: A circuit computed by some code under test.
        expected: The circuit that should have been computed.
    """
    assert actual == expected, (
        "Actual circuit differs from expected circuit.\n"
        "\n"
        "Diagram of actual circuit:\n"
        f"{actual}\n"
        "\n"
        "Diagram of expected circuit:\n"
        f"{expected}\n"
        "\n"
        "Index of first differing moment:\n"
        f"{_first_differing_moment_index(actual, expected)}\n"
        "\n"
        "Full repr of actual circuit:\n"
        f"{actual!r}\n"
        "\n"
        "Full repr of expected circuit:\n"
        f"{expected!r}\n"
    )


def _first_differing_moment_index(
    circuit1: circuits.AbstractCircuit, circuit2: circuits.AbstractCircuit
) -> Optional[int]:
    for i, (m1, m2) in enumerate(itertools.zip_longest(circuit1, circuit2)):
        if m1 != m2:
            return i
    return None  # pragma: no cover


def assert_circuits_have_same_unitary_given_final_permutation(
    actual: circuits.AbstractCircuit,
    expected: circuits.AbstractCircuit,
    qubit_map: Dict[ops.Qid, ops.Qid],
) -> None:
    """Asserts two circuits have the same unitary up to a final permutation of qubits.

    Args:
        actual: A circuit computed by some code under test.
        expected: The circuit that should have been computed.
        qubit_map: the permutation of qubits from the beginning to the end of the circuit.

    Raises:
        ValueError: if 'qubit_map' is not a mapping from the qubits in 'actual' to themselves.
        ValueError: if 'qubit_map' does not have the same set of keys and values.
    """
    if set(qubit_map.keys()) != set(qubit_map.values()):
        raise ValueError("'qubit_map' must have the same set of keys and values.")

    if not set(qubit_map.keys()).issubset(actual.all_qubits()):
        raise ValueError(
            "'qubit_map' must be a mapping of the qubits in the circuit 'actual' to themselves."
        )

    actual_cp = actual.unfreeze()
    initial_qubits, sorted_qubits = zip(*sorted(qubit_map.items(), key=lambda x: x[1]))
    inverse_permutation = [sorted_qubits.index(q) for q in initial_qubits]
    actual_cp.append(ops.QubitPermutationGate(list(inverse_permutation)).on(*sorted_qubits))

    lin_alg_utils.assert_allclose_up_to_global_phase(
        expected.unitary(), actual_cp.unitary(), atol=1e-8
    )


def assert_has_diagram(
    actual: Union[circuits.AbstractCircuit, circuits.Moment], desired: str, **kwargs
) -> None:
    """Determines if a given circuit has the desired text diagram.

    Args:
        actual: The circuit that was actually computed by some process.
        desired: The desired text diagram as a string. Newlines at the
            beginning and whitespace at the end are ignored.
        **kwargs: Keyword arguments to be passed to actual.to_text_diagram().
    """
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable
    actual_diagram = actual.to_text_diagram(**kwargs).lstrip("\n").rstrip()
    desired_diagram = desired.lstrip("\n").rstrip()

    assert actual_diagram == desired_diagram, (
        "Circuit's text diagram differs from the desired diagram.\n"
        '\n'
        'Diagram of actual circuit:\n'
        f'{actual_diagram}\n'
        '\n'
        'Desired text diagram:\n'
        f'{desired_diagram}\n'
        '\n'
        'Highlighted differences:\n'
        f'{highlight_text_differences(actual_diagram, desired_diagram)}\n'
    )


def assert_has_consistent_apply_unitary(val: Any, *, atol: float = 1e-8) -> None:
    """Tests whether a value's _apply_unitary_ is correct.

    Contrasts the effects of the value's `_apply_unitary_` with the
    matrix returned by the value's `_unitary_` method.

    Args:
        val: The value under test. Should have a `__pow__` method.
        atol: Absolute error tolerance.
    """
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    _assert_apply_unitary_works_when_axes_transposed(val, atol=atol)

    expected = protocols.unitary(val, default=None)

    qid_shape = protocols.qid_shape(val)

    eye = qis.eye_tensor((2,) + qid_shape, dtype=np.complex128)
    actual = protocols.apply_unitary(
        unitary_value=val,
        args=protocols.ApplyUnitaryArgs(
            target_tensor=eye,
            available_buffer=np.ones_like(eye) * float('nan'),
            axes=list(range(1, len(qid_shape) + 1)),
        ),
        default=None,
    )

    # If you don't have a unitary, you shouldn't be able to apply a unitary.
    if expected is None:
        assert actual is None
    else:
        expected = np.kron(np.eye(2), expected)

    # If you applied a unitary, it should match the one you say you have.
    if actual is not None:
        assert expected is not None
        n = np.prod([2, *qid_shape])
        np.testing.assert_allclose(actual.reshape(n, n), expected, atol=atol)


def assert_has_consistent_apply_channel(val: Any, *, atol: float = 1e-8) -> None:
    """Tests whether a value's _apply_channel_ is correct.

    Contrasts the effects of the value's `_apply_channel_` with the superoperator calculated from
    the Kraus components returned by the value's `_kraus_` method.

    Args:
        val: The value under test. Should have a `__pow__` method.
        atol: Absolute error tolerance.
    """
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    kraus = protocols.kraus(val, default=None)
    expected = qis.kraus_to_superoperator(kraus) if kraus is not None else None

    qid_shape = protocols.qid_shape(val)

    eye = qis.eye_tensor(qid_shape * 2, dtype=np.complex128)
    actual = protocols.apply_channel(
        val=val,
        args=protocols.ApplyChannelArgs(
            target_tensor=eye,
            out_buffer=np.ones_like(eye) * float('nan'),
            auxiliary_buffer0=np.ones_like(eye) * float('nan'),
            auxiliary_buffer1=np.ones_like(eye) * float('nan'),
            left_axes=list(range(len(qid_shape))),
            right_axes=list(range(len(qid_shape), len(qid_shape) * 2)),
        ),
        default=None,
    )

    # If you don't have a Kraus, you shouldn't be able to apply a channel.
    if expected is None:
        assert actual is None

    # If you applied a channel, it should match the superoperator you say you have.
    if actual is not None:
        assert expected is not None
        n = np.prod(qid_shape) ** 2
        np.testing.assert_allclose(actual.reshape((n, n)), expected, atol=atol)


def _assert_apply_unitary_works_when_axes_transposed(val: Any, *, atol: float = 1e-8) -> None:
    """Tests whether a value's _apply_unitary_ handles out-of-order axes.

    A common mistake to make when implementing `_apply_unitary_` is to assume
    that the incoming axes will be contiguous, or ascending, or that they can be
    flattened, or that other axes have a length of two, etc, etc ,etc. This
    method checks that `_apply_unitary_` does the same thing to out-of-order
    axes that it does to contiguous in-order axes.

    Args:
        val: The operation, gate, or other unitary object to test.
        atol: Absolute error tolerance.

    Raises:
        AssertionError: If `_apply_unitary_` acted differently on the
            out-of-order axes than on the in-order axes.
    """

    # Only test custom apply unitary methods.
    if not hasattr(val, '_apply_unitary_') or not protocols.has_unitary(val):
        return

    # Pick sizes and shapes.
    shape = protocols.qid_shape(val)
    n = len(shape)
    padded_shape = shape + (1, 2, 2, 3)
    padded_n = len(padded_shape)
    size = np.prod(padded_shape, dtype=np.int64).item()

    # Shuffle the axes.
    permutation = list(range(padded_n))
    random.shuffle(permutation)
    transposed_shape = [0] * padded_n
    for i in range(padded_n):
        transposed_shape[permutation[i]] = padded_shape[i]

    # Prepare input states.
    in_order_input = lin_alg_utils.random_superposition(size).reshape(padded_shape)
    out_of_order_input = np.empty(shape=transposed_shape, dtype=np.complex128)
    out_of_order_input.transpose(permutation)[...] = in_order_input

    # Apply to in-order and out-of-order axes.
    in_order_output = protocols.apply_unitary(
        val,
        protocols.ApplyUnitaryArgs(
            target_tensor=in_order_input,
            available_buffer=np.empty_like(in_order_input),
            axes=range(n),
        ),
    )
    out_of_order_output = protocols.apply_unitary(
        val,
        protocols.ApplyUnitaryArgs(
            target_tensor=out_of_order_input,
            available_buffer=np.empty_like(out_of_order_input),
            axes=permutation[:n],
        ),
    )

    # Put the out of order output back into order, to enable comparison.
    reordered_output = out_of_order_output.transpose(permutation)

    # The results should be identical.
    if not np.allclose(in_order_output, reordered_output, atol=atol):
        raise AssertionError(
            f'The _apply_unitary_ method of {repr(val)} acted differently on '
            f'out-of-order axes than on in-order axes.\n'
            f'\n'
            f'The failing axis order: {repr(permutation[:n])}'
        )


def assert_has_consistent_apply_unitary_for_various_exponents(
    val: Any, *, exponents=(0, 1, -1, 0.5, 0.25, -0.5, 0.1, sympy.Symbol('s'))
) -> None:
    """Tests whether a value's _apply_unitary_ is correct.

    Contrasts the effects of the value's `_apply_unitary_` with the
    matrix returned by the value's `_unitary_` method. Attempts this after
    attempting to raise the value to several exponents.

    Args:
        val: The value under test. Should have a `__pow__` method.
        exponents: The exponents to try. Defaults to a variety of special and
            arbitrary angles, as well as a parameterized angle (a symbol). If
            the value's `__pow__` returns `NotImplemented` for any of these,
            they are skipped.
    """
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    for exponent in exponents:
        gate = protocols.pow(val, exponent, default=None)
        if gate is not None:
            assert_has_consistent_apply_unitary(gate)


def assert_has_consistent_qid_shape(val: Any) -> None:
    """Tests whether a value's `_qid_shape_` and `_num_qubits_` are correct and
    consistent.

    Verifies that the entries in the shape are all positive integers and the
    length of shape equals `_num_qubits_` (and also equals `len(qubits)` if
    `val` has `qubits`.

    Args:
        val: The value under test. Should have `_qid_shape_` and/or
            `num_qubits_` methods. Can optionally have a `qubits` property.
    """
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable
    default = (-1,)
    qid_shape = protocols.qid_shape(val, default)
    num_qubits = protocols.num_qubits(val, default)
    if qid_shape is default or num_qubits is default:
        return  # Nothing to check
    assert all(d >= 1 for d in qid_shape), f'Not all entries in qid_shape are positive: {qid_shape}'
    assert (
        len(qid_shape) == num_qubits
    ), f'Length of qid_shape and num_qubits disagree: {qid_shape}, {num_qubits}'

    if isinstance(val, ops.Operation):
        assert num_qubits == len(
            val.qubits
        ), f'Length of num_qubits and val.qubits disagrees: {num_qubits}, {len(val.qubits)}'

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

from typing import Any, Iterable, Optional, Sequence, TYPE_CHECKING, Type

from collections import defaultdict
import itertools
import numpy as np

from cirq import circuits, ops, linalg, protocols, value, EigenGate

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Dict, List


def highlight_text_differences(actual: str, expected: str) -> str:
    diff = ""
    for actual_line, desired_line in itertools.zip_longest(
            actual.splitlines(), expected.splitlines(),
            fillvalue=""):
        diff += "".join(a if a == b else "█"
                        for a, b in itertools.zip_longest(
                            actual_line, desired_line, fillvalue="")) + "\n"
    return diff


def _measurement_subspaces(
        measured_qubits: Iterable[ops.QubitId],
        n_qubits: int
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
    measurement_subspaces = defaultdict(list)  # type: Dict[int, List[int]]
    computational_basis = range(1 << n_qubits)

    for basis_state in computational_basis:
        subspace_key = basis_state & measurement_mask
        measurement_subspaces[subspace_key].append(basis_state)

    subspaces = list(measurement_subspaces.values())

    # Verify this is a partitioning (i.e. full coverage, no overlaps).
    assert sorted(itertools.chain(*subspaces)) == list(computational_basis)

    return subspaces


def assert_circuits_with_terminal_measurements_are_equivalent(
        actual: circuits.Circuit,
        reference: circuits.Circuit,
        atol: float) -> None:
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
    measured_qubits_actual = {qubit
                              for op in actual.all_operations()
                              if ops.MeasurementGate.is_measurement(op)
                              for qubit in op.qubits}
    measured_qubits_reference = {qubit
                                 for op in reference.all_operations()
                                 if ops.MeasurementGate.is_measurement(op)
                                 for qubit in op.qubits}
    assert actual.are_all_measurements_terminal()
    assert reference.are_all_measurements_terminal()
    assert measured_qubits_actual == measured_qubits_reference

    all_qubits = actual.all_qubits().union(reference.all_qubits())

    matrix_actual = actual.to_unitary_matrix(
            qubits_that_should_be_present=all_qubits)
    matrix_reference = reference.to_unitary_matrix(
            qubits_that_should_be_present=all_qubits)

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
        assert linalg.allclose_up_to_global_phase(
                block_actual, block_reference, atol=atol), (
                        "Circuit's effect differs from the reference circuit.\n"
                        '\n'
                        'Diagram of actual circuit:\n'
                        '{}\n'
                        '\n'
                        'Diagram of reference circuit with desired function:\n'
                        '{}\n'.format(actual, reference))


def assert_same_circuits(actual: circuits.Circuit,
                         expected: circuits.Circuit,
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
        "{}\n"
        "\n"
        "Diagram of expected circuit:\n"
        "{}\n"
        "\n"
        "Index of first differing moment:\n"
        "{}\n"
        "\n"
        "Full repr of actual circuit:\n"
        "{!r}\n"
        "\n"
        "Full repr of expected circuit:\n"
        "{!r}\n").format(actual,
                         expected,
                         _first_differing_moment_index(actual, expected),
                         actual,
                         expected)


def _first_differing_moment_index(circuit1: circuits.Circuit,
                                  circuit2: circuits.Circuit) -> Optional[int]:
    for i, (m1, m2) in enumerate(itertools.zip_longest(circuit1, circuit2)):
        if m1 != m2:
            return i
    return None  # coverage: ignore


def assert_has_diagram(
        actual: circuits.Circuit,
        desired: str,
        **kwargs) -> None:
    """Determines if a given circuit has the desired text diagram.

    Args:
        actual: The circuit that was actually computed by some process.
        desired: The desired text diagram as a string. Newlines at the
            beginning and whitespace at the end are ignored.
        **kwargs: Keyword arguments to be passed to actual.to_text_diagram().
    """
    actual_diagram = actual.to_text_diagram(**kwargs).lstrip("\n").rstrip()
    desired_diagram = desired.lstrip("\n").rstrip()
    assert actual_diagram == desired_diagram, (
        "Circuit's text diagram differs from the desired diagram.\n"
        '\n'
        'Diagram of actual circuit:\n'
        '{}\n'
        '\n'
        'Desired text diagram:\n'
        '{}\n'
        '\n'
        'Highlighted differences:\n'
        '{}\n'.format(actual_diagram, desired_diagram,
                      highlight_text_differences(actual_diagram,
                                                 desired_diagram))
    )


def assert_has_consistent_apply_unitary(
        val: Any,
        *,
        qubit_count: Optional[int] = None,
        atol: float=1e-8) -> None:
    """Tests whether a value's _apply_unitary_ is correct.

    Contrasts the effects of the value's `_apply_unitary_` with the
    matrix returned by the value's `_unitary_` method.

    Args:
        val: The value under test. Should have a `__pow__` method.
        qubit_count: Usually inferred. The number of qubits the value acts on.
            This argument isn't needed if the gate has a unitary matrix or
            implements `cirq.SingleQubitGate`/`cirq.TwoQubitGate`/
            `cirq.ThreeQubitGate`.
    """

    expected = protocols.unitary(val, default=None)
    if qubit_count is not None:
        n = qubit_count
    elif expected is not None:
        n = expected.shape[0].bit_length() - 1
    else:
        n = _infer_qubit_count(val)

    eye = np.eye(2 << n, dtype=np.complex128).reshape((2,) * (2 * n + 2))
    actual = protocols.apply_unitary(
        unitary_value=val,
        args=protocols.ApplyUnitaryArgs(
            target_tensor=eye,
            available_buffer=np.ones_like(eye) * float('nan'),
            axes=list(range(1, n + 1))),
        default=None)

    # If you don't have a unitary, you shouldn't be able to apply a unitary.
    if expected is None:
        assert actual is None
    else:
        expected = np.kron(np.eye(2), expected)

    # If you applied a unitary, it should match the one you say you have.
    if actual is not None:
        np.testing.assert_allclose(
            actual.reshape(2 << n, 2 << n),
            expected,
            atol=atol)


def assert_eigen_gate_has_consistent_apply_unitary(
        eigen_gate_type: Type[EigenGate],
        *,
        exponents=(0, 1, -1, 0.5, 0.25, -0.5, 0.1, value.Symbol('s')),
        global_shifts=(0, 0.5, -0.5, 0.1),
        qubit_count: Optional[int] = None) -> None:
    """Tests whether an EigenGate type's _apply_unitary_ is correct.

    Contrasts the effects of the gate's `_apply_unitary_` with the
    matrix returned by the gate's `_unitary_` method, trying various values for
    the gate exponent and global shift.

    Args:
        eigen_gate_type: The type of gate to test. The type must have an
            __init__ method that takes an exponent and a global_shift.
        exponents: The exponents to try. Defaults to a variety of special and
            arbitrary angles, as well as a parameterized angle (a symbol).
        global_shifts: The global shifts to try. Defaults to a variety of
            special angles.
        qubit_count: The qubit count to use for the gate. This argument isn't
            needed if the gate has a unitary matrix or implements
            `cirq.SingleQubitGate`/`cirq.TwoQubitGate`/`cirq.ThreeQubitGate`; it
            will be inferred.
    """
    for exponent in exponents:
        for shift in global_shifts:
            assert_has_consistent_apply_unitary(
                eigen_gate_type(exponent=exponent, global_shift=shift),
                qubit_count=qubit_count)


def assert_has_consistent_apply_unitary_for_various_exponents(
        val: Any,
        *,
        exponents=(0, 1, -1, 0.5, 0.25, -0.5, 0.1, value.Symbol('s')),
        qubit_count: Optional[int] = None) -> None:
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
        qubit_count: A minimum qubit count for the test system. This argument
            isn't needed if the gate has a unitary matrix or implements
            `cirq.SingleQubitGate`/`cirq.TwoQubitGate`/`cirq.ThreeQubitGate`; it
            will be inferred.
    """
    for exponent in exponents:
        gate = protocols.pow(val, exponent, default=None)
        if gate is not None:
            assert_has_consistent_apply_unitary(
                gate,
                qubit_count=qubit_count)


def _infer_qubit_count(val: Any) -> int:
    if isinstance(val, ops.Operation):
        return len(val.qubits)
    if isinstance(val, ops.SingleQubitGate):
        return 1
    if isinstance(val, ops.TwoQubitGate):
        return 2
    if isinstance(val, ops.ThreeQubitGate):
        return 3
    if isinstance(val, ops.ControlledGate):
        return 1 + _infer_qubit_count(val.sub_gate)

    raise NotImplementedError(
        'Failed to infer qubit count of <{!r}>. Specify it.'.format(
            val))

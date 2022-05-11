# Copyright 2022 The Cirq Developers
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

"""Utility methods for decomposing two-qubit unitaries into Sycamore gates."""

from typing import Iterator, List, Optional

import itertools
import math
import numpy as np

import cirq
from cirq_google import ops


def _decompose_arbitrary_into_syc_tabulation(
    op: cirq.Operation, tabulation: cirq.TwoQubitGateTabulation
) -> cirq.OP_TREE:
    """Synthesize an arbitrary 2 qubit operation to a Sycamore operation using the given Tabulation.

    Args:
        op: Operation to decompose.
        tabulation: A `cirq.TwoQubitGateTabulation` for the Sycamore gate.

    Yields:
        A `cirq.OP_TREE` that performs the given operation using Sycamore operations.
    """
    qubit_a, qubit_b = op.qubits
    result = tabulation.compile_two_qubit_gate(cirq.unitary(op))
    local_gates = result.local_unitaries
    for i, (gate_a, gate_b) in enumerate(local_gates):
        yield from _phased_x_z_ops(gate_a, qubit_a)
        yield from _phased_x_z_ops(gate_b, qubit_b)
        if i != len(local_gates) - 1:
            yield ops.SYC.on(qubit_a, qubit_b)


def two_qubit_matrix_to_sycamore_operations(
    q0: cirq.Qid,
    q1: cirq.Qid,
    mat: np.ndarray,
    *,
    atol: float = 1e-8,
    clean_operations: bool = True,
) -> cirq.OP_TREE:
    """Decomposes a two-qubit unitary matrix into `cirq_google.SYC` + single qubit rotations.

    The analytical decomposition first Synthesizes the given operation using `cirq.CZPowGate` +
    single qubit rotations and then decomposes each `cirq.CZPowGate` into `cirq_google.SYC` +
    single qubit rotations using `cirq_google.known_2q_op_to_sycamore_operations`.

    Note that the resulting decomposition may not be optimal, and users should first try to
    decompose a given operation using `cirq_google.known_2q_op_to_sycamore_operations`.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        clean_operations: Merges runs of single qubit gates to a single `cirq.PhasedXZGate` in
            the resulting operations list.

    Returns:
        A `cirq.OP_TREE` that implements the given unitary operation using only `cirq_google.SYC` +
        single qubit rotations.
    """
    decomposed_ops: List[cirq.OP_TREE] = []
    for op in cirq.two_qubit_matrix_to_cz_operations(
        q0, q1, mat, allow_partial_czs=True, atol=atol, clean_operations=clean_operations
    ):
        if cirq.num_qubits(op) == 2:
            decomposed_cphase = known_2q_op_to_sycamore_operations(op)
            assert decomposed_cphase is not None
            decomposed_ops.append(decomposed_cphase)
        else:
            decomposed_ops.append(op)
    return (
        [*cirq.merge_single_qubit_gates_to_phxz(cirq.Circuit(decomposed_ops)).all_operations()]
        if clean_operations
        else decomposed_ops
    )


def known_2q_op_to_sycamore_operations(op: cirq.Operation) -> Optional[cirq.OP_TREE]:
    """Synthesizes a known two-qubit operation using `cirq_google.SYC` + single qubit rotations.

    This function dispatches to various known gate decompositions based on gate type. Currently,
    the following gates are known:
        1. Adjacent `cirq.SWAP` and `cirq.ZPowGate` wrapped in a circuit operation of length 2.
        2. `cirq.PhasedISwapPowGate` with exponent = 1 or phase_exponent = 0.25.
        3. `cirq.SWAP`, `cirq.ISWAP`.
        4. `cirq.CNotPowGate`, `cirq.CZPowGate`, `cirq.ZZPowGate`.

    Args:
        op: Operation to decompose.

    Returns:
        - A `cirq.OP_TREE` that implements the given known operation using only `cirq_google.SYC` +
        single qubit rotations OR
        - None if `op` is not a known operation.
    """
    if not (cirq.has_unitary(op) and cirq.num_qubits(op) == 2):
        return None

    q0, q1 = op.qubits

    if isinstance(op.untagged, cirq.CircuitOperation):
        flattened_gates = [o.gate for o in cirq.decompose_once(op.untagged)]
        if len(flattened_gates) != 2:
            return None
        for g1, g2 in itertools.permutations(flattened_gates):
            if g1 == cirq.SWAP and isinstance(g2, cirq.ZZPowGate):
                return _swap_rzz(g2.exponent * np.pi / 2, q0, q1)

    gate = op.gate
    if isinstance(gate, cirq.PhasedISwapPowGate):
        if math.isclose(gate.exponent, 1) and isinstance(gate.phase_exponent, float):
            return _decompose_phased_iswap_into_syc(gate.phase_exponent, q0, q1)
        if math.isclose(gate.phase_exponent, 0.25):
            return _decompose_phased_iswap_into_syc_precomputed(gate.exponent * np.pi / 2, q0, q1)
        return None
    if isinstance(gate, cirq.CNotPowGate):
        return [
            cirq.Y(q1) ** -0.5,
            _decompose_cphase_into_syc(gate.exponent * np.pi, q0, q1),
            cirq.Y(q1) ** 0.5,
        ]
    if isinstance(gate, cirq.CZPowGate):
        return (
            _decompose_cz_into_syc(q0, q1)
            if math.isclose(gate.exponent, 1)
            else _decompose_cphase_into_syc(gate.exponent * np.pi, q0, q1)
        )
    if isinstance(gate, cirq.SwapPowGate) and math.isclose(gate.exponent, 1):
        return _decompose_swap_into_syc(q0, q1)
    if isinstance(gate, cirq.ISwapPowGate) and math.isclose(gate.exponent, 1):
        return _decompose_iswap_into_syc(q0, q1)
    if isinstance(gate, cirq.ZZPowGate):
        return _rzz(gate.exponent * np.pi / 2, q0, q1)

    return None


def _decompose_phased_iswap_into_syc(
    phase_exponent: float, a: cirq.Qid, b: cirq.Qid
) -> cirq.OP_TREE:
    """Decomposes `cirq.PhasedISwapPowGate` with an exponent of 1 into Sycamore gates.

    This should only be called if the gate has an exponent of 1. Otherwise,
    `_decompose_phased_iswap_into_syc_precomputed` should be used instead. The advantage of using
    this function is that the resulting circuit will be smaller.

    Args:
        phase_exponent: The exponent on the Z gates.
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.PhasedISwapPowGate` gate using Sycamore gates.
    """

    yield cirq.Z(a) ** phase_exponent,
    yield cirq.Z(b) ** -phase_exponent,
    yield _decompose_iswap_into_syc(a, b),
    yield cirq.Z(a) ** -phase_exponent,
    yield cirq.Z(b) ** phase_exponent,


def _decompose_phased_iswap_into_syc_precomputed(
    theta: float, a: cirq.Qid, b: cirq.Qid
) -> cirq.OP_TREE:
    """Decomposes `cirq.PhasedISwapPowGate` into Sycamore gates using precomputed coefficients.

    This should only be called if the Gate has a phase_exponent of .25. If the gate has an
    exponent of 1, _decompose_phased_iswap_into_syc should be used instead. Converting PhasedISwap
    gates to Sycamore is not supported if neither of these constraints are satisfied.

    This synthesize a PhasedISwap in terms of four sycamore gates.  This compilation converts the
    gate into a circuit involving two CZ gates, which themselves are each represented as two
    Sycamore gates and single-qubit rotations

    Args:
        theta: Rotation parameter for the phased ISWAP.
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.PhasedISwapPowGate` gate using Sycamore gates.
    """

    yield cirq.PhasedXPowGate(phase_exponent=0.41175161497166024, exponent=0.5653807577895922).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=1.0, exponent=0.5).on(b),
    yield (cirq.Z**0.7099892314883478).on(b),
    yield (cirq.Z**0.6746023442550453).on(a),
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield ops.SYC(a, b),
    yield cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(a)
    yield (cirq.Z**-0.9255092746611595).on(b)
    yield (cirq.Z**-1.333333333333333).on(a)
    yield cirq.rx(-theta).on(a)
    yield cirq.rx(-theta).on(b)

    yield cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(b)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.8151665352515929, exponent=0.8906746535691492).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.07449072533884049, exponent=0.5).on(b)
    yield (cirq.Z**-0.9255092746611595).on(b)
    yield (cirq.Z**-0.9777346353961884).on(a)


def _decompose_cz_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decomposes `cirq.CZ` into sycamore gates using precomputed coefficients.

    This should only be called when exponent of `cirq.CZPowGate` is 1. Otherwise,
    `_decompose_cphase_into_syc` should be called.

    Args:
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.CZ` gate using Sycamore gates.
    """
    yield cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(b)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield ops.SYC(a, b),
    yield cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(
        a
    ),
    yield (cirq.Z**-0.9255092746611595).on(b),
    yield (cirq.Z**-1.333333333333333).on(a),


def _decompose_cphase_into_syc(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """Implements a cphase using the Ising gate generated from 2 Sycamore gates.

    A cphase gate has the matrix diag([1, 1, 1, exp(1j * theta)]) and can be mapped to the Rzz
    Ising gate +  single qubit Z rotations. We drop the global phase shift of theta / 4.

    Args:
        theta: The phase to apply, exp(1j * theta).
        q0: First qubit to operate on.
        q1: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the cphase gate using Sycamore gates.
    """
    yield _rzz(-theta / 4, q0, q1)
    yield cirq.rz(theta / 2).on(q0)
    yield cirq.rz(theta / 2).on(q1)


def _decompose_iswap_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decomposes `cirq.ISWAP` into sycamore gates using precomputed coefficients.

    This should only be called when exponent of `cirq.ISwapPowGate` is 1. Other cases are currently
    not supported.

    Args:
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.ISWAP` gate using Sycamore gates.
    """
    yield cirq.PhasedXPowGate(phase_exponent=-0.27250925776964596, exponent=0.2893438375555899).on(
        a
    )
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=0.8487591858680898, exponent=0.9749387200813147).on(b),
    yield cirq.PhasedXPowGate(phase_exponent=-0.3582574564210601).on(a),
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=0.9675022326694225, exponent=0.6908986856555526).on(a),
    yield ops.SYC(a, b),
    yield cirq.PhasedXPowGate(phase_exponent=0.9161706861686068, exponent=0.14818318325264102).on(
        b
    ),
    yield cirq.PhasedXPowGate(phase_exponent=0.9408341606787907).on(a),
    yield (cirq.Z**-1.1551880579397293).on(b),
    yield (cirq.Z**0.31848343246696365).on(a),


def _decompose_swap_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decomposes `cirq.SWAP` into sycamore gates using precomputed coefficients.

    This should only be called when exponent of `cirq.SwapPowGate` is 1. Other cases are currently
    not supported.

    Args:
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.SWAP` gate using Sycamore gates.
    """
    yield cirq.PhasedXPowGate(phase_exponent=0.44650378384076217, exponent=0.8817921214052824).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.7656774060816165, exponent=0.6628666504604785).on(b)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.6277589946716742, exponent=0.5659160932099687).on(a)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=0.28890767199499257, exponent=0.4340839067900317).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(a)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.4691261557936808, exponent=0.7728525693920243).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.8150261316932077, exponent=0.11820787859471782).on(
        b
    )
    yield (cirq.Z**-0.7384700844660306).on(b)
    yield (cirq.Z**-0.7034535141382525).on(a)


def _find_local_equivalents(target_unitary: np.ndarray, source_unitary: np.ndarray):
    """Determine the local 1q rotations that turn one equivalent 2q unitary into the other.

    Given two 2q unitaries with the same interaction coefficients but different local unitary
    rotations determine the local unitaries that turns one type of gate into another.

    1) Perform the KAK Decomposition on each unitary and confirm interaction terms are equivalent.
    2) Identify the elements of SU(2) to transform source_unitary into target_unitary

    Args:
        target_unitary: The unitary that we need to transform `source_unitary` to.
        source_unitary: The unitary that we need to transform by adding local gates, and make it
            equivalent to the target_unitary.

    Returns:
        Four 2x2 unitaries. The first two are pre-rotations and last two are post rotations.
    """
    kak_u1 = cirq.kak_decomposition(target_unitary)
    kak_u2 = cirq.kak_decomposition(source_unitary)

    u_0 = kak_u1.single_qubit_operations_after[0] @ kak_u2.single_qubit_operations_after[0].conj().T
    u_1 = kak_u1.single_qubit_operations_after[1] @ kak_u2.single_qubit_operations_after[1].conj().T

    v_0 = (
        kak_u2.single_qubit_operations_before[0].conj().T @ kak_u1.single_qubit_operations_before[0]
    )
    v_1 = (
        kak_u2.single_qubit_operations_before[1].conj().T @ kak_u1.single_qubit_operations_before[1]
    )

    return v_0, v_1, u_0, u_1


def _create_corrected_circuit(
    target_unitary: np.ndarray, program: cirq.Circuit, q0: cirq.Qid, q1: cirq.Qid
) -> cirq.OP_TREE:
    """Adds pre/post single qubit rotations to `program` to make it equivalent to `target_unitary`.

    Adds single qubit correction terms to the given circuit on 2 qubit s.t. it implements
    `target_unitary`. This assumes that `program` implements a 2q unitary effect which has same
    interaction coefficients as `target_unitary` in it's KAK decomposition and differs only in
    local unitary rotations.

    Args:
        target_unitary: The unitary that should be implemented by the transformed `program`.
        program: `cirq.Circuit` to be transformed.
        q0: First qubit to operate on.
        q1: Second qubit to operate on.

    Yields:
        Operations in `program` with pre and post rotations added s.t. the resulting `cirq.OP_TREE`
        implements `target_unitary`.
    """
    # Get the local equivalents
    b_0, b_1, a_0, a_1 = _find_local_equivalents(
        target_unitary, program.unitary(qubit_order=cirq.QubitOrder.explicit([q0, q1]))
    )

    # Apply initial corrections
    yield from _phased_x_z_ops(b_0, q0)
    yield from _phased_x_z_ops(b_1, q1)

    # Apply interaction part
    yield program

    # Apply final corrections
    yield from _phased_x_z_ops(a_0, q0)
    yield from _phased_x_z_ops(a_1, q1)


def _phased_x_z_ops(mat: np.ndarray, q: cirq.Qid) -> Iterator[cirq.Operation]:
    """Yields `cirq.PhasedXZGate` operation implementing `mat` if it is not identity."""
    gate = cirq.single_qubit_matrix_to_phxz(mat)
    if gate:
        yield gate(q)


def _rzz(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """Implements the Rzz Ising coupling gate (i.e. exp(-1j * theta * zz)) using Sycamore gates.

    Args:
        theta: The rotation parameter of Rzz Ising coupling gate.
        q0: First qubit to operate on
        q1: Second qubit to operate on

    Yields:
        The `cirq.OP_TREE` that implements the Rzz Ising coupling gate using Sycamore gates.
    """
    phi = -np.pi / 24
    c_phi = np.cos(2 * phi)
    target_unitary = cirq.unitary(cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5))
    c2 = abs(np.sin(theta) if abs(np.cos(theta)) > c_phi else np.cos(theta)) / c_phi

    # Prepare program that has same Schmidt coefficients as exp(-1j theta ZZ)
    program = cirq.Circuit(ops.SYC(q0, q1), cirq.rx(2 * np.arccos(c2)).on(q1), ops.SYC(q0, q1))

    yield _create_corrected_circuit(target_unitary, program, q0, q1)


def _swap_rzz(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """An implementation of SWAP * exp(-1j * theta * ZZ) using three sycamore gates.

    This builds off of the _rzz method.

    Args:
        theta: The rotation parameter of Rzz Ising coupling gate.
        q0: First qubit to operate on.
        q1: Second qubit to operate on.

    Yields:
        The `cirq.OP_TREE`` that implements ZZ followed by a swap.
    """

    # Set interaction part.
    angle_offset = np.pi / 24 - np.pi / 4
    circuit = cirq.Circuit(ops.SYC(q0, q1), _rzz(theta - angle_offset, q0, q1))

    # Get the intended circuit.
    intended_circuit = cirq.Circuit(
        cirq.SWAP(q0, q1), cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(q0, q1)
    )

    yield _create_corrected_circuit(cirq.unitary(intended_circuit), circuit, q0, q1)

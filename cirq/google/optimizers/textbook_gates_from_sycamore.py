"""
Compilation of the sycamore gate to textbook gates
"""
from typing import cast

from itertools import product
from functools import reduce

import math
import numpy as np
import scipy.linalg
from cirq import circuits, google, linalg, ops, optimizers, protocols

UNITARY_X = protocols.unitary(ops.X)
UNITARY_Y = protocols.unitary(ops.Y)
UNITARY_Z = protocols.unitary(ops.Z)
UNITARY_ZZ = np.kron(UNITARY_Z, UNITARY_Z)


def decompose_cz_into_syc(a: ops.Qid, b: ops.Qid):
    """Decompose CZ into sycamore gates using precomputed coefficients"""
    yield ops.PhasedXPowGate(phase_exponent=0.5678998743900456,
                             exponent=0.5863459345743176).on(a)
    yield ops.PhasedXPowGate(phase_exponent=0.3549946157441739).on(b)
    yield google.SYC.on(a, b)
    yield ops.PhasedXPowGate(phase_exponent=-0.5154334589432878,
                             exponent=0.5228733015013345).on(b)
    yield ops.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield google.SYC.on(a, b),
    yield ops.PhasedXPowGate(phase_exponent=-0.5987667922766213,
                             exponent=0.4136540654256824).on(a),
    yield (ops.Z**-0.9255092746611595).on(b),
    yield (ops.Z**-1.333333333333333).on(a),


def decompose_iswap_into_syc(a: ops.Qid, b: ops.Qid):
    """Decompose ISWAP into sycamore gates using precomputed coefficients"""
    yield ops.PhasedXPowGate(phase_exponent=-0.27250925776964596,
                             exponent=0.2893438375555899).on(a)
    yield google.SYC.on(a, b)
    yield ops.PhasedXPowGate(phase_exponent=0.8487591858680898,
                             exponent=0.9749387200813147).on(b),
    yield ops.PhasedXPowGate(phase_exponent=-0.3582574564210601).on(a),
    yield google.SYC.on(a, b)
    yield ops.PhasedXPowGate(phase_exponent=0.9675022326694225,
                             exponent=0.6908986856555526).on(a),
    yield google.SYC.on(a, b),
    yield ops.PhasedXPowGate(phase_exponent=0.9161706861686068,
                             exponent=0.14818318325264102).on(b),
    yield ops.PhasedXPowGate(phase_exponent=0.9408341606787907).on(a),
    yield (ops.Z**-1.1551880579397293).on(b),
    yield (ops.Z**0.31848343246696365).on(a),


def decompose_swap_into_syc(a: ops.Qid, b: ops.Qid):
    """Decompose SWAP into sycamore gates using precomputed coefficients"""
    yield ops.PhasedXPowGate(phase_exponent=0.44650378384076217,
                             exponent=0.8817921214052824).on(a)
    yield ops.PhasedXPowGate(phase_exponent=-0.7656774060816165,
                             exponent=0.6628666504604785).on(b)
    yield google.SYC.on(a, b)
    yield ops.PhasedXPowGate(phase_exponent=-0.6277589946716742,
                             exponent=0.5659160932099687).on(a)
    yield google.SYC.on(a, b)
    yield ops.PhasedXPowGate(phase_exponent=0.28890767199499257,
                             exponent=0.4340839067900317).on(b)
    yield ops.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(a)
    yield google.SYC.on(a, b)
    yield ops.PhasedXPowGate(phase_exponent=-0.4691261557936808,
                             exponent=0.7728525693920243).on(a)
    yield ops.PhasedXPowGate(phase_exponent=-0.8150261316932077,
                             exponent=0.11820787859471782).on(b)
    yield (ops.Z**-0.7384700844660306).on(b)
    yield (ops.Z**-0.7034535141382525).on(a)


def find_local_equivalents(unitary1: np.ndarray, unitary2: np.ndarray):
    """
    Given two unitaries with the same interaction coefficients but different
    local unitary rotations determine the local unitaries that turns
    one type of gate into another.

    1) perform the kak decomp on each unitary and confirm interaction terms
       are equivalent
    2) identitfy the elements of SU(2) to transform unitary2 into unitary1

    Args:
        unitary1: unitary that we target
        unitary2: unitary that we transform the local gates to the target
    Returns:
        four 2x2 unitaries.  first two are pre-rotations last two are post
        rotations.
    """
    kak_u1 = linalg.kak_decomposition(unitary1)
    kak_u2 = linalg.kak_decomposition(unitary2)

    u_0 = (kak_u1.single_qubit_operations_after[0]
           @ kak_u2.single_qubit_operations_after[0].conj().T)
    u_1 = (kak_u1.single_qubit_operations_after[1]
           @ kak_u2.single_qubit_operations_after[1].conj().T)

    v_0 = (kak_u2.single_qubit_operations_before[0].conj().T
           @ kak_u1.single_qubit_operations_before[0])
    v_1 = (kak_u2.single_qubit_operations_before[1].conj().T
           @ kak_u1.single_qubit_operations_before[1])

    return v_0, v_1, u_0, u_1


def create_corrected_circuit(target_unitary: np.ndarray,
                             program: circuits.Circuit, q0: ops.Qid,
                             q1: ops.Qid):
    # Get the local equivalents
    b_0, b_1, a_0, a_1 = find_local_equivalents(
        target_unitary,
        program.unitary(qubit_order=ops.QubitOrder.explicit([q0, q1])))

    # Apply initial corrections
    yield (gate(q0) for gate in optimizers.single_qubit_matrix_to_gates(b_0))
    yield (gate(q1) for gate in optimizers.single_qubit_matrix_to_gates(b_1))

    # Apply interaction part
    yield program

    # Apply final corrections
    yield (gate(q0) for gate in optimizers.single_qubit_matrix_to_gates(a_0))
    yield (gate(q1) for gate in optimizers.single_qubit_matrix_to_gates(a_1))


def zztheta(theta: float, q0: ops.Qid, q1: ops.Qid) -> ops.OP_TREE:
    """Generate exp(-1j * theta * zz) from Sycamore gates.

    Args:
        theta: rotation parameter
        q0: First qubit id to operate on
        q1: Second qubit id to operate on
    Returns:
        a Cirq program implementing the Ising gate
    rtype:
        cirq.OP_Tree
    """
    phi = -np.pi / 24
    c_phi = np.cos(2 * phi)
    target_unitary = scipy.linalg.expm(-1j * theta * UNITARY_ZZ)

    if abs(np.cos(theta)) > c_phi:
        c2 = abs(np.sin(theta)) / c_phi
    else:
        c2 = abs(np.cos(theta)) / c_phi

    # Prepare program that has same Schmidt coeffs as exp(1j theta ZZ)
    program = circuits.Circuit(google.SYC.on(q0, q1),
                               ops.Rx(2 * np.arccos(c2)).on(q1),
                               google.SYC.on(q0, q1))

    yield create_corrected_circuit(target_unitary, program, q0, q1)


def cphase(theta: float, q0: ops.Qid, q1: ops.Qid) -> ops.OP_TREE:
    """
    Implement a cphase using the Ising gate generated from two Sycamore gates

    A CPHASE gate has the matrix diag([1, 1, 1, exp(1j * theta)]) and can
    be mapped to the Ising gate by prep and post rotations of Z-pi/4.
    We drop the global phase shift of theta/4.

    Args:
        theta: exp(1j * theta )
        q0: First qubit id to operate on
        q1: Second qubit id to operate on
    returns:
        a cirq program implementating cphase
    rtype:
        cirq.OP_TREE
    """
    yield zztheta(-theta / 4, q0, q1)
    yield ops.Rz(theta / 2).on(q0)
    yield ops.Rz(theta / 2).on(q1)


def swap_zztheta(theta: float, q0: ops.Qid, q1: ops.Qid) -> ops.OP_TREE:
    """
    An implementation of

    SWAP * EXP(1j theta ZZ)

    Using three sycamore gates.

    This builds off of the zztheta method.  A sycamore gate following the
    zz-gate is a SWAP EXP(1j (THETA - pi/24) ZZ).

    Args:
        theta: exp(1j * theta )
        q0: First qubit id to operate on
        q1: Second qubit id to operate on
    Returns:
        The circuit that implements ZZ followed by a swap

    :rtype cirq.OP_TREE
    """

    # Set interaction part.
    circuit = circuits.Circuit()
    angle_offset = np.pi / 24 - np.pi / 4
    circuit.append(google.SYC(q0, q1))
    circuit.append(zztheta(theta - angle_offset, q0, q1))

    # Get the intended circuit.
    intended_circuit = circuits.Circuit(
        ops.SWAP(q0, q1),
        ops.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(q0, q1))

    yield create_corrected_circuit(intended_circuit, circuit, q0, q1)


def operator_decomp(operator):
    pauli_ops = [np.eye(2), UNITARY_X, UNITARY_Y, UNITARY_Z]
    num_qubits = int(np.log2(operator.shape[0]))
    coeff_vector = np.zeros(4**num_qubits, dtype=np.complex128)
    for idx, vec_index in enumerate(product(range(4), repeat=num_qubits)):
        op_basis = reduce(np.kron, map(lambda x: pauli_ops[x], vec_index))
        assert np.allclose(
            np.kron(pauli_ops[vec_index[0]], pauli_ops[vec_index[1]]), op_basis)
        coeff_vector[idx] = np.trace(op_basis.conj().T.dot(operator))

    coeff_vector /= 2**num_qubits

    return coeff_vector


def known_two_q_operations_to_sycamore_operations(qubit_a: ops.Qid,
                                                  qubit_b: ops.Qid,
                                                  op: ops.GateOperation
                                                 ) -> ops.OP_TREE:
    """
    Synthesis a known gate operation to a sycamore operation

    This function dispatches based on gate type

    Args:
        qubit_a: first qubit of GateOperation
        qubit_b: second qubit of GateOperation
        op:
    Returns:
        New operations iterable object
    """
    gate = op.gate
    if isinstance(gate, ops.CNotPowGate):
        return [
            ops.Y(qubit_b)**-0.5,
            cphase(
                cast(ops.CNotPowGate, gate).exponent * np.pi, qubit_a, qubit_b),
            ops.Y(qubit_b)**0.5,
        ]
    elif isinstance(gate, ops.CZPowGate):
        gate = cast(ops.CZPowGate, gate)
        if math.isclose(gate.exponent, 1.0):  # check if CZ or CPHASE
            return decompose_cz_into_syc(qubit_a, qubit_b)
        else:
            # because CZPowGate == diag([1, 1, 1, e^{i pi phi}])
            return cphase(gate.exponent * np.pi, qubit_a, qubit_b)
    elif isinstance(gate, ops.SwapPowGate) and math.isclose(
            cast(ops.SwapPowGate, gate).exponent, 1.0):
        return decompose_swap_into_syc(qubit_a, qubit_b)
    elif isinstance(gate, ops.ISwapPowGate) and math.isclose(
            cast(ops.ISwapPowGate, gate).exponent, 1.0):
        return decompose_iswap_into_syc(qubit_a, qubit_b)
    elif isinstance(gate, ops.ZZPowGate):
        return zztheta(
            cast(ops.ZZPowGate, gate).exponent * np.pi / 2, *op.qubits)
    elif isinstance(gate, ops.MatrixGate) and len(op.qubits) == 2:
        new_ops = optimizers.two_qubit_matrix_to_operations(
            op.qubits[0], op.qubits[1], op, allow_partial_czs=True)
        gate_ops = []
        for new_op in new_ops:
            num_qubits = len(new_op.qubits)
            if num_qubits == 1:
                gate_ops.extend([
                    term.on(new_op.qubits[0])
                    for term in optimizers.single_qubit_matrix_to_gates(
                        protocols.unitary(new_op))
                ])
            elif num_qubits == 2:
                gate_ops.extend(
                    ops.flatten_to_ops(
                        known_two_q_operations_to_sycamore_operations(
                            new_op.qubits[0], new_op.qubits[1],
                            cast(ops.GateOperation, new_op))))
        return gate_ops
    else:
        raise ValueError("Unrecognized gate: {!r}".format(op))

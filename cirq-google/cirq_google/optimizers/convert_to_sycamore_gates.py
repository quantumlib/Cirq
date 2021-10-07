# Copyright 2019 The Cirq Developers
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
import math
from typing import Iterator, List, Optional

import numpy as np
import scipy.linalg

import cirq
import cirq_google as google
from cirq_google.ops import SycamoreGate
from cirq_google.optimizers.two_qubit_gates.gate_compilation import GateTabulation


UNITARY_ZZ = np.kron(cirq.unitary(cirq.Z), cirq.unitary(cirq.Z))
PAULI_OPS = [
    np.eye(2),
    cirq.unitary(cirq.X),
    cirq.unitary(cirq.Y),
    cirq.unitary(cirq.Z),
]


class ConvertToSycamoreGates(cirq.PointOptimizer):
    """Attempts to convert non-native gates into SycamoreGates.

    First, checks if the given operation is already a native sycamore operation.

    Second, checks if the operation has a known unitary. If so, and the gate
        is a 1-qubit or 2-qubit gate, then performs circuit synthesis of the
        operation.

    Third, attempts to `cirq.decompose` to the operation.

    Fourth, if ignore_failures is set, gives up and returns the gate unchanged.
        Otherwise raises a TypeError.
    """

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def __init__(self, tabulation: Optional[GateTabulation] = None, ignore_failures=False) -> None:
        """Inits ConvertToSycamoreGates.

        Args:
            tabulation: If set, a tabulation for the Sycamore gate to use for
                decomposing Matrix gates. If unset, an analytic calculation is
                used for Matrix gates. To get a GateTabulation, call the
                gate_product_tabulation method with a base gate (in this case,
                usually cirq_google.SYC) and a maximum infidelity.
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
        """
        super().__init__()
        self.ignore_failures = ignore_failures
        if tabulation is not None and not isinstance(tabulation, GateTabulation):
            raise ValueError("provided tabulation must be of type GateTabulation")
        self.tabulation = tabulation

    # pylint: enable=missing-raises-doc
    def _is_native_sycamore_op(self, op: cirq.Operation) -> bool:
        """Check if the given operation is native to a Sycamore device.

        Args:
            op: Input operation.

        Returns:
            True if the operation is native to the gmon, false otherwise.
        """
        gate = op.gate

        if isinstance(
            gate,
            (
                SycamoreGate,
                cirq.MeasurementGate,
                cirq.PhasedXZGate,
                cirq.PhasedXPowGate,
                cirq.XPowGate,
                cirq.YPowGate,
                cirq.ZPowGate,
            ),
        ):
            return True

        if (
            isinstance(gate, cirq.FSimGate)
            and math.isclose(gate.theta, np.pi / 2)
            and math.isclose(gate.phi, np.pi / 6)
        ):
            return True

        return False

    # TODO(#3388) Add summary line to docstring.
    # pylint: disable=docstring-first-line-empty
    def _convert_one(self, op: cirq.Operation) -> cirq.OP_TREE:
        """
        Decomposer intercept:  Upon cirq.protocols.decompose catch and
        return new OP_Tree

        This should decompose based on number of qubits.
        """
        if len(op.qubits) == 1:
            return _phased_x_z_ops(cirq.unitary(op, None), op.qubits[0])
        elif len(op.qubits) == 2 and isinstance(op, cirq.GateOperation):
            return known_two_q_operations_to_sycamore_operations(
                op.qubits[0], op.qubits[1], op, self.tabulation
            )
        return NotImplemented

    # pylint: enable=docstring-first-line-empty
    def convert(self, op: cirq.Operation) -> List[cirq.Operation]:
        def on_stuck_raise(bad):
            return TypeError(
                "Don't know how to work with {!r}. "
                "It isn't a native xmon operation, "
                "a 1 or 2 qubit gate with a known unitary, "
                "or composite.".format(bad)
            )

        return cirq.decompose(
            op,
            keep=self._is_native_sycamore_op,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=None if self.ignore_failures else on_stuck_raise,
        )

    def optimization_at(
        self, circuit: cirq.Circuit, index: int, op: cirq.Operation
    ) -> Optional[cirq.PointOptimizationSummary]:
        if not isinstance(op, cirq.GateOperation):
            return None

        gate = op.gate

        # Check for a SWAP and ZZPowGate together
        if isinstance(gate, cirq.ZZPowGate) or gate == cirq.SWAP:
            gate2 = None
            rads = None
            next_index = circuit.next_moment_operating_on(op.qubits, index + 1)
            if next_index is not None:
                ops_in_front = list({circuit.operation_at(q, next_index) for q in op.qubits})
                if len(ops_in_front) == 1 and isinstance(ops_in_front[0], cirq.GateOperation):
                    gate2 = ops_in_front[0].gate
            else:
                next_index = 0

            if isinstance(gate, cirq.SwapPowGate) and isinstance(gate2, cirq.ZZPowGate):
                rads = gate2.exponent * np.pi / 2
            if isinstance(gate, cirq.ZZPowGate) and gate2 == cirq.SWAP:
                rads = gate.exponent * np.pi / 2
            if rads is not None:
                return cirq.PointOptimizationSummary(
                    clear_span=next_index - index + 1,
                    clear_qubits=op.qubits,
                    new_operations=swap_rzz(rads, op.qubits[0], op.qubits[1]),
                )

        converted = self.convert(op)
        if len(converted) == 1 and converted[0] is op:
            return None

        return cirq.PointOptimizationSummary(
            clear_span=1, new_operations=converted, clear_qubits=op.qubits
        )


# TODO(#3388) Add documentation for Raises.
# pylint: disable=missing-raises-doc
def known_two_q_operations_to_sycamore_operations(
    qubit_a: cirq.Qid,
    qubit_b: cirq.Qid,
    op: cirq.Operation,
    tabulation: Optional[GateTabulation] = None,
) -> cirq.OP_TREE:
    """Synthesizes a known gate operation to a sycamore operation.

    This function dispatches based on gate type

    Args:
        qubit_a: first qubit of GateOperation
        qubit_b: second qubit of GateOperation
        op: operation to decompose
        tabulation: A tabulation for the Sycamore gate to use for
            decomposing gates.
    Returns:
        New operations iterable object
    """
    gate = op.gate
    if isinstance(gate, cirq.PhasedISwapPowGate):
        if math.isclose(gate.exponent, 1):
            return decompose_phased_iswap_into_syc(gate.phase_exponent, qubit_a, qubit_b)
        elif math.isclose(gate.phase_exponent, 0.25):
            return decompose_phased_iswap_into_syc_precomputed(
                gate.exponent * np.pi / 2, qubit_a, qubit_b
            )
        else:
            raise ValueError(
                "To decompose PhasedISwapPowGate, it must have a phase_exponent"
                " of .25 OR an exponent of 1.0, but got: {!r}".format(op)
            )
    if isinstance(gate, cirq.CNotPowGate):
        return [
            cirq.Y(qubit_b) ** -0.5,
            cphase(gate.exponent * np.pi, qubit_a, qubit_b),
            cirq.Y(qubit_b) ** 0.5,
        ]
    elif isinstance(gate, cirq.CZPowGate):
        if math.isclose(gate.exponent, 1):  # check if CZ or CPHASE
            return decompose_cz_into_syc(qubit_a, qubit_b)
        else:
            # because CZPowGate == diag([1, 1, 1, e^{i pi phi}])
            return cphase(gate.exponent * np.pi, qubit_a, qubit_b)
    elif isinstance(gate, cirq.SwapPowGate) and math.isclose(gate.exponent, 1):
        return decompose_swap_into_syc(qubit_a, qubit_b)
    elif isinstance(gate, cirq.ISwapPowGate) and math.isclose(gate.exponent, 1):
        return decompose_iswap_into_syc(qubit_a, qubit_b)
    elif isinstance(gate, cirq.ZZPowGate):
        return rzz(gate.exponent * np.pi / 2, *op.qubits)
    elif cirq.unitary(gate, None) is not None:
        if tabulation:
            return decompose_arbitrary_into_syc_tabulation(qubit_a, qubit_b, op, tabulation)
        else:
            return decompose_arbitrary_into_syc_analytic(qubit_a, qubit_b, op)
    else:
        raise ValueError(f"Unrecognized gate: {op!r}")


# pylint: enable=missing-raises-doc
def decompose_phased_iswap_into_syc(
    phase_exponent: float, a: cirq.Qid, b: cirq.Qid
) -> cirq.OP_TREE:
    """Decompose PhasedISwap with an exponent of 1.

    This should only be called if the Gate has an exponent of 1 - otherwise,
    decompose_phased_iswap_into_syc_precomputed should be used instead. The
    advantage of using this function is that the resulting circuit will be
    smaller.

    Args:
        phase_exponent: The exponent on the Z gates.
        a: First qubit id to operate on
        b: Second qubit id to operate on
    Returns:
        a Cirq program implementing the Phased ISWAP gate

    """

    yield cirq.Z(a) ** phase_exponent,
    yield cirq.Z(b) ** -phase_exponent,
    yield decompose_iswap_into_syc(a, b),
    yield cirq.Z(a) ** -phase_exponent,
    yield cirq.Z(b) ** phase_exponent,


def decompose_phased_iswap_into_syc_precomputed(
    theta: float, a: cirq.Qid, b: cirq.Qid
) -> cirq.OP_TREE:
    """Decompose PhasedISwap into sycamore gates using precomputed coefficients.

    This should only be called if the Gate has a phase_exponent of .25. If the
    gate has an exponent of 1, decompose_phased_iswap_into_syc should be used
    instead. Converting PhasedISwap gates to Sycamore is not supported if
    neither of these constraints are satisfied.

    This synthesize a PhasedISwap in terms of four sycamore gates.  This
    compilation converts the gate into a circuit involving two CZ gates, which
    themselves are each represented as two Sycamore gates and single-qubit
    rotations

    Args:
        theta: rotation parameter
        a: First qubit id to operate on
        b: Second qubit id to operate on
    Returns:
        a Cirq program implementing the Phased ISWAP gate

    """

    yield cirq.PhasedXPowGate(phase_exponent=0.41175161497166024, exponent=0.5653807577895922).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=1.0, exponent=0.5).on(b),
    yield (cirq.Z ** 0.7099892314883478).on(b),
    yield (cirq.Z ** 0.6746023442550453).on(a),
    yield SycamoreGate().on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield SycamoreGate().on(a, b),
    yield cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(a)
    yield (cirq.Z ** -0.9255092746611595).on(b)
    yield (cirq.Z ** -1.333333333333333).on(a)
    yield cirq.rx(-theta).on(a)
    yield cirq.rx(-theta).on(b)

    yield cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(b)
    yield SycamoreGate().on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield SycamoreGate().on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.8151665352515929, exponent=0.8906746535691492).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.07449072533884049, exponent=0.5).on(b)
    yield (cirq.Z ** -0.9255092746611595).on(b)
    yield (cirq.Z ** -0.9777346353961884).on(a)


def decompose_arbitrary_into_syc_tabulation(
    qubit_a: cirq.Qid, qubit_b: cirq.Qid, op: cirq.Operation, tabulation: GateTabulation
) -> cirq.OP_TREE:
    """Synthesize an arbitrary 2 qubit operation to a sycamore operation using
    the given Tabulation.

    Args:
        qubit_a: first qubit of the operation
        qubit_b: second qubit of the operation
        op: operation to decompose
        tabulation: A tabulation for the Sycamore gate.
    Returns:
        New operations iterable object
    """
    result = tabulation.compile_two_qubit_gate(cirq.unitary(op))
    local_gates = result.local_unitaries
    for i, (gate_a, gate_b) in enumerate(local_gates):
        yield from _phased_x_z_ops(gate_a, qubit_a)
        yield from _phased_x_z_ops(gate_b, qubit_b)
        if i != len(local_gates) - 1:
            yield google.SYC.on(qubit_a, qubit_b)


def decompose_arbitrary_into_syc_analytic(
    qubit_a: cirq.Qid, qubit_b: cirq.Qid, op: cirq.Operation
) -> cirq.OP_TREE:
    """Synthesize an arbitrary 2 qubit operation to a sycamore operation using
    the given Tabulation.

     Args:
            qubit_a: first qubit of the operation
            qubit_b: second qubit of the operation
            op: operation to decompose
            tabulation: A tabulation for the Sycamore gate.
        Returns:
            New operations iterable object
    """
    new_ops = cirq.two_qubit_matrix_to_operations(qubit_a, qubit_b, op, allow_partial_czs=True)
    for new_op in new_ops:
        num_qubits = len(new_op.qubits)
        if num_qubits == 1:
            (a,) = new_op.qubits
            yield from _phased_x_z_ops(cirq.unitary(new_op), a)
        elif num_qubits == 2:
            a, b = op.qubits
            yield from cirq.flatten_to_ops(
                known_two_q_operations_to_sycamore_operations(a, b, new_op)
            )


def decompose_cz_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decompose CZ into sycamore gates using precomputed coefficients"""
    yield cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(b)
    yield google.SYC.on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield google.SYC.on(a, b),
    yield cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(
        a
    ),
    yield (cirq.Z ** -0.9255092746611595).on(b),
    yield (cirq.Z ** -1.333333333333333).on(a),


def decompose_iswap_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decompose ISWAP into sycamore gates using precomputed coefficients"""
    yield cirq.PhasedXPowGate(phase_exponent=-0.27250925776964596, exponent=0.2893438375555899).on(
        a
    )
    yield google.SYC.on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=0.8487591858680898, exponent=0.9749387200813147).on(b),
    yield cirq.PhasedXPowGate(phase_exponent=-0.3582574564210601).on(a),
    yield google.SYC.on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=0.9675022326694225, exponent=0.6908986856555526).on(a),
    yield google.SYC.on(a, b),
    yield cirq.PhasedXPowGate(phase_exponent=0.9161706861686068, exponent=0.14818318325264102).on(
        b
    ),
    yield cirq.PhasedXPowGate(phase_exponent=0.9408341606787907).on(a),
    yield (cirq.Z ** -1.1551880579397293).on(b),
    yield (cirq.Z ** 0.31848343246696365).on(a),


def decompose_swap_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decompose SWAP into sycamore gates using precomputed coefficients"""
    yield cirq.PhasedXPowGate(phase_exponent=0.44650378384076217, exponent=0.8817921214052824).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.7656774060816165, exponent=0.6628666504604785).on(b)
    yield google.SYC.on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.6277589946716742, exponent=0.5659160932099687).on(a)
    yield google.SYC.on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=0.28890767199499257, exponent=0.4340839067900317).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.22592784059288928).on(a)
    yield google.SYC.on(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.4691261557936808, exponent=0.7728525693920243).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=-0.8150261316932077, exponent=0.11820787859471782).on(
        b
    )
    yield (cirq.Z ** -0.7384700844660306).on(b)
    yield (cirq.Z ** -0.7034535141382525).on(a)


# TODO(#3388) Add summary line to docstring.
# pylint: disable=docstring-first-line-empty
def find_local_equivalents(unitary1: np.ndarray, unitary2: np.ndarray):
    """
    Given two unitaries with the same interaction coefficients but different
    local unitary rotations determine the local unitaries that turns
    one type of gate into another.

    1) perform the kak decomp on each unitary and confirm interaction terms
       are equivalent
    2) identify the elements of SU(2) to transform unitary2 into unitary1

    Args:
        unitary1: unitary that we target
        unitary2: unitary that we transform the local gates to the target
    Returns:
        four 2x2 unitaries.  first two are pre-rotations last two are post
        rotations.
    """
    kak_u1 = cirq.kak_decomposition(unitary1)
    kak_u2 = cirq.kak_decomposition(unitary2)

    u_0 = kak_u1.single_qubit_operations_after[0] @ kak_u2.single_qubit_operations_after[0].conj().T
    u_1 = kak_u1.single_qubit_operations_after[1] @ kak_u2.single_qubit_operations_after[1].conj().T

    v_0 = (
        kak_u2.single_qubit_operations_before[0].conj().T @ kak_u1.single_qubit_operations_before[0]
    )
    v_1 = (
        kak_u2.single_qubit_operations_before[1].conj().T @ kak_u1.single_qubit_operations_before[1]
    )

    return v_0, v_1, u_0, u_1


# pylint: enable=docstring-first-line-empty
def create_corrected_circuit(
    target_unitary: np.ndarray, program: cirq.Circuit, q0: cirq.Qid, q1: cirq.Qid
) -> cirq.OP_TREE:
    # Get the local equivalents
    b_0, b_1, a_0, a_1 = find_local_equivalents(
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
    gate = cirq.single_qubit_matrix_to_phxz(mat)
    if gate:
        yield gate(q)


def rzz(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
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
    program = cirq.Circuit(
        google.SYC.on(q0, q1), cirq.rx(2 * np.arccos(c2)).on(q1), google.SYC.on(q0, q1)
    )

    yield create_corrected_circuit(target_unitary, program, q0, q1)


def cphase(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """Implements a cphase using the Ising gate generated from 2 Sycamore gates.

    A CPHASE gate has the matrix diag([1, 1, 1, exp(1j * theta)]) and can
    be mapped to the Ising gate by prep and post rotations of Z-pi/4.
    We drop the global phase shift of theta/4.

    Args:
        theta: exp(1j * theta )
        q0: First qubit id to operate on
        q1: Second qubit id to operate on
    returns:
        a cirq program implementing cphase
    """
    yield rzz(-theta / 4, q0, q1)
    yield cirq.rz(theta / 2).on(q0)
    yield cirq.rz(theta / 2).on(q1)


def swap_rzz(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """An implementation of SWAP * EXP(1j theta ZZ) using three sycamore gates.

    This builds off of the zztheta method.  A sycamore gate following the
    zz-gate is a SWAP EXP(1j (THETA - pi/24) ZZ).

    Args:
        theta: exp(1j * theta )
        q0: First qubit id to operate on
        q1: Second qubit id to operate on
    Returns:
        The circuit that implements ZZ followed by a swap
    """

    # Set interaction part.
    circuit = cirq.Circuit()
    angle_offset = np.pi / 24 - np.pi / 4
    circuit.append(google.SYC(q0, q1))
    circuit.append(rzz(theta - angle_offset, q0, q1))

    # Get the intended circuit.
    intended_circuit = cirq.Circuit(
        cirq.SWAP(q0, q1), cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(q0, q1)
    )

    yield create_corrected_circuit(intended_circuit, circuit, q0, q1)

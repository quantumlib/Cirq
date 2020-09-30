from typing import (
    Sequence,
    Union,
    Any,
    List,
    Iterator,
    TYPE_CHECKING,
    Iterable,
    Optional,
)

import numpy as np

from cirq import ops, linalg, circuits, devices
from cirq.optimizers import merge_single_qubit_gates, drop_empty_moments
from cirq._compat import deprecated

if TYPE_CHECKING:
    import cirq


@deprecated(deadline='v0.10',
            fix='Use cirq.decompose_two_qubit_interaction_into_four_fsim_gates.'
           )
def decompose_two_qubit_interaction_into_four_fsim_gates_via_b(
        interaction: Union['cirq.Operation', 'cirq.Gate', np.ndarray, Any],
        *,
        fsim_gate: Union['cirq.FSimGate', 'cirq.ISwapPowGate'],
        qubits: Sequence['cirq.Qid'] = None) -> 'cirq.Circuit':
    circuit = decompose_two_qubit_interaction_into_four_fsim_gates(
        interaction, fsim_gate=fsim_gate, qubits=qubits)
    merge_single_qubit_gates.MergeSingleQubitGates().optimize_circuit(circuit)
    drop_empty_moments.DropEmptyMoments().optimize_circuit(circuit)
    return circuit


def decompose_two_qubit_interaction_into_four_fsim_gates(
        interaction: Union['cirq.Operation', 'cirq.Gate', np.ndarray, Any],
        *,
        fsim_gate: Union['cirq.FSimGate', 'cirq.ISwapPowGate'],
        qubits: Sequence['cirq.Qid'] = None) -> 'cirq.Circuit':
    """Decomposes operations into an FSimGate near theta=pi/2, phi=0.

    This decomposition is guaranteed to use exactly four of the given FSim
    gates. It works by decomposing into two B gates and then decomposing each
    B gate into two of the given FSim gate.

    This decomposition only works for FSim gates with a theta (iswap angle)
    between 3/8π and 5/8π (i.e. within 22.5° of maximum strength) and a
    phi (cphase angle) between -π/4 and +π/4 (i.e. within 45° of minimum
    strength).

    Args:
        interaction: The two qubit operation to synthesize. This can either be
            a cirq object (such as a gate, operation, or circuit) or a raw numpy
            array specifying the 4x4 unitary matrix.
        fsim_gate: The only two qubit gate that is permitted to appear in the
            output. Must satisfy 3/8π < phi < 5/8π and abs(theta) < pi/4.
        qubits: The qubits that the resulting operations should apply the
            desired interaction to. If not set then defaults to either the
            qubits of the given interaction (if it is a `cirq.Operation`) or
            else to `cirq.LineQubit.range(2)`.

    Returns:
        A list of operations implementing the desired two qubit unitary. The
        list will include four operations of the given fsim gate, various single
        qubit operations, and a global phase operation.
    """
    if isinstance(fsim_gate, ops.ISwapPowGate):
        mapped_gate = ops.FSimGate(-fsim_gate.exponent * np.pi / 2, 0)
    else:
        mapped_gate = fsim_gate
    if not 3 / 8 * np.pi <= abs(mapped_gate.theta) <= 5 / 8 * np.pi:
        raise ValueError('Must have 3π/8 ≤ |fsim_gate.theta| ≤ 5π/8')
    if abs(mapped_gate.phi) > np.pi / 4:
        raise ValueError('Must have abs(fsim_gate.phi) ≤ π/4')
    if qubits is None:
        if isinstance(interaction, ops.Operation):
            qubits = interaction.qubits
        else:
            qubits = devices.LineQubit.range(2)
    if len(qubits) != 2:
        raise ValueError(f'Expected a pair of qubits, but got {qubits!r}.')
    kak = linalg.kak_decomposition(interaction)

    result_using_b_gates = _decompose_two_qubit_interaction_into_two_b_gates(
        kak, qubits=qubits)

    b_decomposition = _decompose_b_gate_into_two_fsims(fsim_gate=mapped_gate,
                                                       qubits=qubits)
    b_decomposition = [
        fsim_gate(*op.qubits) if op.gate == mapped_gate else op
        for op in b_decomposition
    ]

    result = circuits.Circuit()
    for op in result_using_b_gates:
        if isinstance(op.gate, _BGate):
            result.append(b_decomposition)
        else:
            result.append(op)
    return result


def _sticky_0_to_1(v: float, *, atol: float) -> Optional[float]:
    if 0 <= v <= 1:
        return v
    if 1 < v <= 1 + atol:
        return 1
    if 0 > v >= -atol:
        return 0
    return None


def _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(
        *,
        qubits: Sequence['cirq.Qid'],
        fsim_gate: 'cirq.FSimGate',
        canonical_x_kak_coefficient: float,
        canonical_y_kak_coefficient: float,
        atol: float = 1e-8) -> List['cirq.Operation']:
    x = canonical_x_kak_coefficient
    y = canonical_y_kak_coefficient
    assert 0 <= y <= x <= np.pi / 4

    eta = np.sin(x)**2 * np.cos(y)**2 + np.cos(x)**2 * np.sin(y)**2
    xi = abs(np.sin(2 * x) * np.sin(2 * y))

    t = fsim_gate.phi / 2
    kappa = np.sin(fsim_gate.theta)**2 - np.sin(t)**2
    s_sum = (eta - np.sin(t)**2) / kappa
    s_dif = 0.5 * xi / kappa

    a_dif = _sticky_0_to_1(s_sum + s_dif, atol=atol)
    a_sum = _sticky_0_to_1(s_sum - s_dif, atol=atol)
    if a_dif is None or a_sum is None:
        raise ValueError(
            f'Failed to synthesize XX^{x/np.pi}·YY^{y/np.pi} from two '
            f'{fsim_gate!r} separated by single qubit operations.')

    x_dif = np.arcsin(np.sqrt(a_dif))
    x_sum = np.arcsin(np.sqrt(a_sum))

    x_a = x_sum + x_dif
    x_b = x_dif - x_sum

    a, b = qubits
    return [
        fsim_gate(a, b),
        ops.rz(t + np.pi).on(a),
        ops.rz(t).on(b),
        ops.rx(x_a).on(a),
        ops.rx(x_b).on(b),
        fsim_gate(a, b),
    ]


class _BGate(ops.Gate):
    """Single qubit gates and two of these can achieve any kak coefficients.

    References:
        Minimum construction of two-qubit quantum operations
        https://arxiv.org/abs/quant-ph/0312193
    """

    def num_qubits(self) -> int:
        return 2

    def _decompose_(self, qubits):
        a, b = qubits
        return [
            ops.XX(a, b)**-0.5,
            ops.YY(a, b)**-0.25,
        ]


_B = _BGate()


def _decompose_two_qubit_interaction_into_two_b_gates(
        interaction: Union['cirq.Operation', 'cirq.Gate', np.ndarray, Any], *,
        qubits: Sequence['cirq.Qid']) -> List['cirq.Operation']:
    kak = linalg.kak_decomposition(interaction)

    result = _decompose_interaction_into_two_b_gates_ignoring_single_qubit_ops(
        qubits, kak.interaction_coefficients)

    return list(
        _fix_single_qubit_gates_around_kak_interaction(desired=kak,
                                                       qubits=qubits,
                                                       operations=result))


def _decompose_b_gate_into_two_fsims(*, fsim_gate: 'cirq.FSimGate',
                                     qubits: Sequence['cirq.Qid']
                                    ) -> List['cirq.Operation']:
    kak = linalg.kak_decomposition(_B)

    result = _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(
        qubits=qubits,
        fsim_gate=fsim_gate,
        canonical_x_kak_coefficient=kak.interaction_coefficients[0],
        canonical_y_kak_coefficient=kak.interaction_coefficients[1])

    return list(
        _fix_single_qubit_gates_around_kak_interaction(desired=kak,
                                                       qubits=qubits,
                                                       operations=result))


def _decompose_interaction_into_two_b_gates_ignoring_single_qubit_ops(
        qubits: Sequence['cirq.Qid'],
        kak_interaction_coefficients: Iterable[float]
) -> List['cirq.Operation']:
    """
    References:
        Minimum construction of two-qubit quantum operations
        https://arxiv.org/abs/quant-ph/0312193
    """
    a, b = qubits
    x, y, z = kak_interaction_coefficients
    r = (np.sin(y) * np.cos(z))**2
    r = max(0.0, r)  # Clamp out-of-range floating point error.
    if r > 0.499999999999:
        rb = [
            ops.ry(np.pi).on(b),
        ]
    else:
        b1 = np.cos(y * 2) * np.cos(z * 2) / (1 - 2 * r)
        b1 = max(0.0, min(1, b1))  # Clamp out-of-range floating point error.
        b2 = np.arcsin(np.sqrt(b1))
        b3 = np.arccos(1 - 4 * r)
        rb = [
            ops.rz(-b2).on(b),
            ops.ry(-b3).on(b),
            ops.rz(-b2).on(b),
        ]
    s = 1 if z < 0 else -1
    return [
        _B(a, b),
        ops.ry(s * 2 * x).on(a),
        *rb,
        _B(a, b),
    ]


def _fix_single_qubit_gates_around_kak_interaction(
        *,
        desired: 'cirq.KakDecomposition',
        operations: List['cirq.Operation'],
        qubits: Sequence['cirq.Qid'],
) -> Iterator['cirq.Operation']:
    """Adds single qubit operations to complete a desired interaction.

    Args:
        desired: The kak decomposition of the desired operation.
        qubits: The pair of qubits that is being operated on.
        operations: A list of operations that composes into the desired kak
            interaction coefficients, but may not have the desired before/after
            single qubit operations or the desired global phase.

    Returns:
        A list of operations whose kak decomposition approximately equals the
        desired kak decomposition.
    """
    actual = linalg.kak_decomposition(
        circuits.Circuit(operations).unitary(qubit_order=qubits))

    def dag(a: np.ndarray) -> np.ndarray:
        return np.transpose(np.conjugate(a))

    for k in range(2):
        g = ops.MatrixGate(
            dag(actual.single_qubit_operations_before[k])
            @ desired.single_qubit_operations_before[k])
        yield g(qubits[k])
    yield from operations
    for k in range(2):
        g = ops.MatrixGate(desired.single_qubit_operations_after[k] @ dag(
            actual.single_qubit_operations_after[k]))
        yield g(qubits[k])
    yield ops.GlobalPhaseOperation(desired.global_phase / actual.global_phase)

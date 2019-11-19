from typing import Sequence, Union, Any, List, Iterator, TYPE_CHECKING, Iterable

import numpy as np

from cirq import ops, linalg, circuits, devices
from cirq.optimizers import merge_single_qubit_gates, drop_empty_moments

if TYPE_CHECKING:
    import cirq


def decompose_two_qubit_interaction_into_four_fsim_gates_via_b(
        interaction: Union['cirq.Operation', 'cirq.Gate', np.ndarray, Any],
        *,
        fsim_gate: 'cirq.FSimGate',
        qubits: Sequence['cirq.Qid'] = None) -> 'cirq.Circuit':
    """Decomposes operations into an FSimGate near theta=pi/2, phi=0.

    This decomposition is guaranteed to use exactly four of the given FSim
    gates. It works by decomposing into two B gates and then decomposing each
    B gate into two of the given FSim gate.

    TODO: describe the feasible angles.

    Args:
        interaction: The two qubit operation to synthesize. This can either be
            a cirq object (such as a gate, operation, or circuit) or a raw numpy
            array specifying the 4x4 unitary matrix.
        fsim_gate: The only two qubit gate that is permitted to appear in the
            output.
        qubits: The qubits that the resulting operations should apply the
            desired interaction to. If not set then defaults to either the
            qubits of the given interaction (if it is a `cirq.Operation`) or
            else to `cirq.LineQubit.range(2)`.

    Returns:
        A list of operations implementing the desired two qubit unitary. The
        list will include four operations of the given fsim gate, various single
        qubit operations, and a global phase operation.
    """
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

    b_decomposition = _decompose_b_gate_into_two_fsims(fsim_gate=fsim_gate,
                                                       qubits=qubits)
    result = []
    for op in result_using_b_gates:
        if isinstance(op.gate, _BGate):
            result.extend(b_decomposition)
        else:
            result.append(op)

    circuit = circuits.Circuit(result)
    merge_single_qubit_gates.MergeSingleQubitGates().optimize_circuit(circuit)
    drop_empty_moments.DropEmptyMoments().optimize_circuit(circuit)
    return circuit


def _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(
        *, qubits: Sequence['cirq.Qid'], fsim_gate: 'cirq.FSimGate',
        canonical_x_kak_coefficient: float,
        canonical_y_kak_coefficient: float) -> List['cirq.Operation']:
    x = canonical_x_kak_coefficient
    y = canonical_y_kak_coefficient
    assert 0 <= y <= x <= np.pi / 4

    eta = np.sin(x)**2 * np.cos(y)**2 + np.cos(x)**2 * np.sin(y)**2
    xi = abs(np.sin(2 * x) * np.sin(2 * y))

    t = fsim_gate.phi / 2
    kappa = np.sin(fsim_gate.theta)**2 - np.sin(t)**2
    s_sum = (eta - np.sin(t)**2) / kappa
    s_dif = 0.5 * xi / kappa

    x_dif = np.arcsin(np.sqrt(s_sum + s_dif))
    x_sum = np.arcsin(np.sqrt(s_sum - s_dif))

    x_a = x_sum + x_dif
    x_b = x_dif - x_sum
    if np.isnan(x_b):
        raise ValueError(
            f'Failed to synthesize XX^{x/np.pi}·YY^{y/np.pi} from two '
            f'{fsim_gate!r} separated by single qubit operations.')

    a, b = qubits
    return [
        fsim_gate(a, b),
        ops.Rz(t + np.pi).on(a),
        ops.Rz(t).on(b),
        ops.Rx(x_a).on(a),
        ops.Rx(x_b).on(b),
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
            ops.XX(a, b)**0.5,
            ops.YY(a, b)**0.25,
        ]

    def __str__(self):
        return 'B'


def _decompose_two_qubit_interaction_into_two_b_gates(
        interaction: Union['cirq.Operation', 'cirq.Gate', np.ndarray, Any],
        *,
        qubits: Sequence['cirq.Qid'] = None) -> List['cirq.Operation']:
    if qubits is None:
        if isinstance(interaction, ops.Operation):
            qubits = interaction.qubits
        else:
            qubits = devices.LineQubit.range(2)
    if len(qubits) != 2:
        raise ValueError(f'Expected a pair of qubits, but got {qubits!r}.')
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
    kak = linalg.kak_decomposition(_BGate())

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
        kak_interaction_coefficients: Iterable[float]) -> List['cirq.Operation']:
    """
    References:
        Minimum construction of two-qubit quantum operations
        https://arxiv.org/abs/quant-ph/0312193
    """
    a, b = qubits
    x, y, z = kak_interaction_coefficients
    B = _BGate()
    r = (np.sin(y) * np.cos(z))**2
    b1 = np.arccos(1 - 4 * r)
    b2 = np.arcsin(np.sqrt(np.cos(y * 2) * np.cos(z * 2) / (1 - 2 * r)))
    s = 1 if z < 0 else -1
    return [
        B(a, b),
        ops.Ry(s * 2 * x).on(a),
        ops.Rz(b2).on(b),
        ops.Ry(b1).on(b),
        ops.Rz(b2).on(b),
        B(a, b),
    ]


def _fix_single_qubit_gates_around_kak_interaction(
        desired: 'cirq.KakDecomposition', qubits: Sequence['cirq.Qid'],
        operations: List['cirq.Operation']) -> Iterator['cirq.Operation']:
    actual = linalg.kak_decomposition(circuits.Circuit(operations))

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

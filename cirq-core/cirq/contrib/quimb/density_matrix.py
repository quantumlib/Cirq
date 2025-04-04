# pylint: disable=wrong-or-nonexistent-copyright-notice
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import quimb
import quimb.tensor as qtn

import cirq


@lru_cache()
def _qpos_tag(qubits: Union[cirq.Qid, Tuple[cirq.Qid]]):
    """Given a qubit or qubits, return a "position tag" (used for drawing).

    For multiple qubits, the tag is for the first qubit.
    """
    if isinstance(qubits, cirq.Qid):
        return _qpos_tag((qubits,))
    x = min(qubits)
    return f'q{x}'


@lru_cache()
def _qpos_y(
    qubits: Union[cirq.Qid, Tuple[cirq.Qid, ...]], all_qubits: Tuple[cirq.Qid, ...]
) -> float:
    """Given a qubit or qubits, return the position y value (used for drawing).

    For multiple qubits, the position is the mean of the qubit indices.
    This "flips" the coordinate so qubit 0 is at the maximal y position.

    Args:
        qubits: The qubits involved in the tensor.
        all_qubits: All qubits in the circuit, allowing us
            to position the zero'th qubit at the top.
    """
    if isinstance(qubits, cirq.Qid):
        return _qpos_y((qubits,), all_qubits)

    pos = [all_qubits.index(q) for q in qubits]
    x = np.mean(pos).item()
    return len(all_qubits) - x - 1


def _add_to_positions(
    positions: Dict[Tuple[str, str], Tuple[float, float]],
    mi: int,
    qubits: Union[cirq.Qid, Tuple[cirq.Qid]],
    *,
    all_qubits: Tuple[cirq.Qid, ...],
    x_scale,
    y_scale,
    x_nudge,
    yb_offset,
):
    """Helper function to update the `positions` dictionary.

    Args:
        positions: The dictionary to update. Quimb will consume this for drawing
        mi: Moment index (used for x-positioning)
        qubits: The qubits (used for y-positioning)
        all_qubits: All qubits in the circuit, allowing us
            to position the zero'th qubit at the top.
        x_scale: Stretch coordinates in the x direction
        y_scale: Stretch coordinates in the y direction
        x_nudge: Kraus operators will have vertical lines connecting the
            "forward" and "backward" circuits, so the x position of each
            tensor is nudged (according to its y position) to help see all
            the lines.
        yb_offset: Offset the "backwards" circuit by this much.
    """
    qy = _qpos_y(qubits, all_qubits)
    positions[(f'i{mi}f', _qpos_tag(qubits))] = (mi * x_scale + qy * x_nudge, y_scale * qy)
    positions[(f'i{mi}b', _qpos_tag(qubits))] = (mi * x_scale, y_scale * qy + yb_offset)


def circuit_to_density_matrix_tensors(
    circuit: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]] = None
) -> Tuple[List[qtn.Tensor], Dict['cirq.Qid', int], Dict[Tuple[str, str], Tuple[float, float]]]:
    """Given a circuit with mixtures or channels, construct a tensor network
    representation of the density matrix.

    This assumes you start in the |0..0><0..0| state. Indices are named
    "nf{i}_q{x}" and "nb{i}_q{x}" where i is a time index and x is a
    qubit index. nf- and nb- refer to the "forwards" and "backwards"
    copies of the circuit. Kraus indices are named "k{j}" where j is an
    independent "kraus" internal index which you probably never need to access.

    Args:
        circuit: The circuit containing operations that support the
            cirq.unitary() or cirq.kraus() protocols.
        qubits: The qubits in the circuit. The `positions` return argument
            will position qubits according to their index in this list.

    Returns:
        tensors: A list of Quimb Tensor objects
        qubit_frontier: A mapping from qubit to time index at the end of
            the circuit. This can be used to deduce the names of the free
            tensor indices.
        positions: A positions dictionary suitable for passing to tn.graph()'s
            `fix` argument to draw the resulting tensor network similar to a
            quantum circuit.

    Raises:
        ValueError: If an op is encountered that cannot be converted.
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())  # pragma: no cover
    qubits = tuple(qubits)

    qubit_frontier: Dict[cirq.Qid, int] = {q: 0 for q in qubits}
    kraus_frontier = 0
    positions: Dict[Tuple[str, str], Tuple[float, float]] = {}
    tensors: List[qtn.Tensor] = []

    x_scale = 2
    y_scale = 3
    x_nudge = 0.3
    n_qubits = len(qubits)
    yb_offset = (n_qubits + 0.5) * y_scale

    def _positions(_mi, _these_qubits):
        return _add_to_positions(
            positions,
            _mi,
            _these_qubits,
            all_qubits=qubits,
            x_scale=x_scale,
            y_scale=y_scale,
            x_nudge=x_nudge,
            yb_offset=yb_offset,
        )

    # Initialize forwards and backwards qubits into the 0 state, i.e. prepare
    # rho_0 = |0><0|.
    for q in qubits:
        tensors += [
            qtn.Tensor(
                data=quimb.up().squeeze(), inds=(f'nf0_q{q}',), tags={'Q0', 'i0f', _qpos_tag(q)}
            ),
            qtn.Tensor(
                data=quimb.up().squeeze(), inds=(f'nb0_q{q}',), tags={'Q0', 'i0b', _qpos_tag(q)}
            ),
        ]
        _positions(0, q)

    for mi, moment in enumerate(circuit.moments):
        for op in moment.operations:
            start_inds_f = [f'nf{qubit_frontier[q]}_q{q}' for q in op.qubits]
            start_inds_b = [f'nb{qubit_frontier[q]}_q{q}' for q in op.qubits]
            for q in op.qubits:
                qubit_frontier[q] += 1
            end_inds_f = [f'nf{qubit_frontier[q]}_q{q}' for q in op.qubits]
            end_inds_b = [f'nb{qubit_frontier[q]}_q{q}' for q in op.qubits]

            if cirq.has_unitary(op):
                U = cirq.unitary(op).reshape((2,) * 2 * len(op.qubits)).astype(np.complex128)
                tensors.append(
                    qtn.Tensor(
                        data=U,
                        inds=end_inds_f + start_inds_f,
                        tags={f'Q{len(op.qubits)}', f'i{mi + 1}f', _qpos_tag(op.qubits)},
                    )
                )
                tensors.append(
                    qtn.Tensor(
                        data=np.conj(U),
                        inds=end_inds_b + start_inds_b,
                        tags={f'Q{len(op.qubits)}', f'i{mi + 1}b', _qpos_tag(op.qubits)},
                    )
                )
            elif cirq.has_kraus(op):
                K = np.asarray(cirq.kraus(op), dtype=np.complex128)
                kraus_inds = [f'k{kraus_frontier}']
                tensors.append(
                    qtn.Tensor(
                        data=K,
                        inds=kraus_inds + end_inds_f + start_inds_f,
                        tags={f'kQ{len(op.qubits)}', f'i{mi + 1}f', _qpos_tag(op.qubits)},
                    )
                )
                tensors.append(
                    qtn.Tensor(
                        data=np.conj(K),
                        inds=kraus_inds + end_inds_b + start_inds_b,
                        tags={f'kQ{len(op.qubits)}', f'i{mi + 1}b', _qpos_tag(op.qubits)},
                    )
                )
                kraus_frontier += 1
            else:
                raise ValueError(repr(op))  # pragma: no cover

            _positions(mi + 1, op.qubits)
    return tensors, qubit_frontier, positions


def tensor_density_matrix(
    circuit: cirq.Circuit, qubits: Optional[List[cirq.Qid]] = None
) -> np.ndarray:
    """Given a circuit with mixtures or channels, contract a tensor network
    representing the resultant density matrix.

    Note: If the circuit contains 6 qubits or fewer, we use a bespoke
    contraction ordering that corresponds to the "normal" in-time contraction
    ordering. Otherwise, the contraction order determination could take
    longer than doing the contraction. Your mileage may vary and benchmarking
    is encouraged for your particular problem if performance is important.
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())

    tensors, qubit_frontier, _ = circuit_to_density_matrix_tensors(circuit=circuit, qubits=qubits)
    tn = qtn.TensorNetwork(tensors)
    f_inds = tuple(f'nf{qubit_frontier[q]}_q{q}' for q in qubits)
    b_inds = tuple(f'nb{qubit_frontier[q]}_q{q}' for q in qubits)
    if len(qubits) <= 6:
        # Heuristic: don't try to determine best order for low qubit number
        # Just contract in time.
        tags_seq = [(f'i{i}b', f'i{i}f') for i in range(len(circuit) + 1)]
        tn.contract_cumulative(tags_seq, inplace=True)
    else:
        tn.contract(inplace=True)
    return tn.to_dense(f_inds, b_inds)

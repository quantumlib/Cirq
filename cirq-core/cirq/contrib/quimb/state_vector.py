# pylint: disable=wrong-or-nonexistent-copyright-notice
import warnings
from typing import cast, Sequence, Union, List, Tuple, Dict, Optional

import numpy as np
import quimb
import quimb.tensor as qtn

import cirq


# coverage: ignore
def _get_quimb_version():
    """Returns the quimb version and parsed (major,minor) numbers if possible.
    Returns:
        a tuple of ((major, minor), version string)
    """
    version = quimb.__version__
    try:
        return tuple(int(x) for x in version.split('.')), version
    except:
        return (0, 0), version


QUIMB_VERSION = _get_quimb_version()


def circuit_to_tensors(
    circuit: cirq.Circuit,
    qubits: Optional[Sequence[cirq.Qid]] = None,
    initial_state: Union[int, None] = 0,
) -> Tuple[List[qtn.Tensor], Dict['cirq.Qid', int], None]:
    """Given a circuit, construct a tensor network representation.

    Indices are named "i{i}_q{x}" where i is a time index and x is a
    qubit index.

    Args:
        circuit: The circuit containing operations that implement the
            cirq.unitary() protocol.
        qubits: A list of qubits in the circuit.
        initial_state: Either `0` corresponding to the |0..0> state, in
            which case the tensor network will represent the final
            state vector; or `None` in which case the starting indices
            will be left open and the tensor network will represent the
            circuit unitary.
    Returns:
        tensors: A list of quimb Tensor objects
        qubit_frontier: A mapping from qubit to time index at the end of
            the circuit. This can be used to deduce the names of the free
            tensor indices.
        positions: Currently None. May be changed in the future to return
            a suitable mapping for tn.graph()'s `fix` argument. Currently,
            `fix=None` will draw the resulting tensor network using a spring
            layout.

    Raises:
        ValueError: If the ihitial state is anything other than that
            corresponding to the |0> state.
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())  # coverage: ignore

    qubit_frontier = {q: 0 for q in qubits}
    positions = None
    tensors: List[qtn.Tensor] = []

    if initial_state == 0:
        for q in qubits:
            tensors += [qtn.Tensor(data=quimb.up().squeeze(), inds=(f'i0_q{q}',), tags={'Q0'})]
    elif initial_state is None:
        # no input tensors, return a network representing the unitary
        pass
    else:
        raise ValueError("Right now, only |0> or `None` initial states are supported.")

    for moment in circuit.moments:
        for op in moment.operations:
            assert cirq.has_unitary(op.gate)
            start_inds = [f'i{qubit_frontier[q]}_q{q}' for q in op.qubits]
            for q in op.qubits:
                qubit_frontier[q] += 1
            end_inds = [f'i{qubit_frontier[q]}_q{q}' for q in op.qubits]

            U = cirq.unitary(op).reshape((2,) * 2 * len(op.qubits))
            t = qtn.Tensor(data=U, inds=end_inds + start_inds, tags={f'Q{len(op.qubits)}'})
            tensors.append(t)

    return tensors, qubit_frontier, positions


def tensor_state_vector(
    circuit: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]] = None
) -> np.ndarray:
    """Given a circuit contract a tensor network into a final state vector."""
    if qubits is None:
        qubits = sorted(circuit.all_qubits())

    tensors, qubit_frontier, _ = circuit_to_tensors(circuit=circuit, qubits=qubits)
    tn = qtn.TensorNetwork(tensors)
    f_inds = tuple(f'i{qubit_frontier[q]}_q{q}' for q in qubits)
    tn.contract(inplace=True)
    return tn.to_dense(f_inds)


def tensor_unitary(
    circuit: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]] = None
) -> np.ndarray:
    """Given a circuit contract a tensor network into a dense unitary
    of the circuit."""
    if qubits is None:
        qubits = sorted(circuit.all_qubits())

    tensors, qubit_frontier, _ = circuit_to_tensors(
        circuit=circuit, qubits=qubits, initial_state=None
    )
    tn = qtn.TensorNetwork(tensors)
    i_inds = tuple(f'i0_q{q}' for q in qubits)
    f_inds = tuple(f'i{qubit_frontier[q]}_q{q}' for q in qubits)
    tn.contract(inplace=True)
    return tn.to_dense(f_inds, i_inds)


def circuit_for_expectation_value(
    circuit: cirq.Circuit, pauli_string: cirq.PauliString
) -> cirq.Circuit:
    """Sandwich a PauliString operator between a forwards and backwards
    copy of a circuit.

    This is a circuit representation of the expectation value of an operator
    <A> = <psi|A|psi> = <0|U^dag A U|0>. You can either extract the 0..0
    amplitude of the final state vector (assuming starting from the |0..0>
    state or extract the [0, 0] entry of the unitary matrix of this combined
    circuit.
    """
    assert pauli_string.coefficient == 1
    return cirq.Circuit(
        [
            circuit,
            cirq.Moment(gate.on(q) for q, gate in pauli_string.items()),
            cirq.inverse(circuit),
        ]
    )


def tensor_expectation_value(
    circuit: cirq.Circuit, pauli_string: cirq.PauliString, max_ram_gb=16, tol=1e-6
) -> float:
    """Compute an expectation value for an operator and a circuit via tensor
    contraction.

    This will give up if it looks like the computation will take too much RAM.
    """
    circuit_sand = circuit_for_expectation_value(circuit, pauli_string / pauli_string.coefficient)
    qubits = sorted(circuit_sand.all_qubits())

    tensors, qubit_frontier, _ = circuit_to_tensors(circuit=circuit_sand, qubits=qubits)
    end_bras = [
        qtn.Tensor(
            data=quimb.up().squeeze(), inds=(f'i{qubit_frontier[q]}_q{q}',), tags={'Q0', 'bra0'}
        )
        for q in qubits
    ]
    tn = qtn.TensorNetwork(tensors + end_bras)
    if QUIMB_VERSION[0] < (1, 3):
        # coverage: ignore
        warnings.warn(
            f'quimb version {QUIMB_VERSION[1]} detected. Please use '
            f'quimb>=1.3 for optimal performance in '
            '`tensor_expectation_value`. '
            'See https://github.com/quantumlib/Cirq/issues/3263'
        )
    else:
        tn.rank_simplify(inplace=True)
    path_info = tn.contract(get='path-info')
    ram_gb = path_info.largest_intermediate * 128 / 8 / 1024 / 1024 / 1024
    if ram_gb > max_ram_gb:
        raise MemoryError(f"We estimate that this contraction will take too much RAM! {ram_gb} GB")
    e_val = tn.contract(inplace=True)
    assert e_val.imag < tol
    assert cast(complex, pauli_string.coefficient).imag < tol
    return e_val.real * pauli_string.coefficient

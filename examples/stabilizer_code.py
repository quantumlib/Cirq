import numpy as np
from typing import List

import cirq

# Based on:
# Stabilizer Codes and Quantum Error Correction
# Daniel Gottesman
# https://thesis.library.caltech.edu/2900/2/THESIS.pdf


def _BuildByCode(mat: np.ndarray) -> List[str]:
    """
    Takes into input a matrix of Boolean interpreted as row-vectors, each having dimension 2 * n.
    The matrix is converted into another matrix with as many rows, but this time the vectors
    contain the letters I, X, Y, and Z representing Pauli operators.
    """
    out = []
    n = mat.shape[1] // 2
    for i in range(mat.shape[0]):
        ps = ''
        for j in range(n):
            if mat[i, j] == 0 and mat[i, j + n] == 0:
                ps += 'I'
            elif mat[i, j] == 1 and mat[i, j + n] == 0:
                ps += 'X'
            elif mat[i, j] == 0 and mat[i, j + n] == 1:
                ps += 'Z'
            else:
                ps += 'Y'
        out.append(ps)
    return out


# It was considered to use scipy.linalg.lu but it seems to be only for real numbers and does
# not allow to restrict only on a section of the matrix.
def _GaussianElimination(
    M: np.ndarray, min_row: int, max_row: int, min_col: int, max_col: int
) -> int:
    """
    Performs a Gaussian elemination of the input matrix and transforms it into its reduced row
    echelon form. The elimination is done only on a sub-section of the matrix (specified) by
    ranges of rows and columns. The matrix elements are integers {0, 1} interpreted as elements
    of GF(2).

    In short, this is the implementation of section 4.1 of the thesis.

    Args:
        M: The input/output matrix
        min_row: The minimum row (inclusive) where the perform the elimination.
        max_row: The maximum row (exclusive) where the perform the elimination.
        min_col: The minimum column (inclusive) where the perform the elimination.
        max_col: The maximum column (exclusive) where the perform the elimination.

    Returns:
        The rank of the matrix.
    """
    max_rank = min(max_row - min_row, max_col - min_col)

    rank = 0
    for r in range(max_rank):
        i = min_row + r
        j = min_col + r
        pivot_rows, pivot_cols = np.nonzero(M[i:max_row, j:max_col])

        if pivot_rows.size == 0:
            break

        pi = pivot_rows[0]
        pj = pivot_cols[0]

        # Swap the rows and columns:
        M[[i, i + pi]] = M[[i + pi, i]]
        M[:, [(j + pj), j]] = M[:, [j, (j + pj)]]

        # Do the elimination.
        for k in range(i + 1, max_row):
            if M[k, j] == 1:
                M[k, :] = np.mod(M[i, :] + M[k, :], 2)

        rank += 1

    # Backward replacing to get identity
    for r in reversed(range(rank)):
        i = min_row + r
        j = min_col + r

        # Do the elimination.
        for k in reversed(range(min_row, i)):
            if M[k, j] == 1:
                M[k, :] = np.mod(M[i, :] + M[k, :], 2)

    return rank


class StabilizerCode(object):
    def __init__(self, group_generators: List[str], allowed_errors: List[str]):
        n = len(group_generators[0])
        k = n - len(group_generators)

        # Build the matrix defined in section 3.4. Each row corresponds to one generator of the
        # code, which is a vector of dimension n. The elements of the vectors are Pauli matrices
        # encoded as I, X, Y, or Z. However, as described in the thesis, we encode the Pauli
        # vector of 2*n Booleans.
        M = np.zeros((n - k, 2 * n), np.int8)
        for i, group_generator in enumerate(group_generators):
            for j, c in enumerate(group_generator):
                if c == 'X' or c == 'Y':
                    M[i, j] = 1
                elif c == 'Z' or c == 'Y':
                    M[i, n + j] = 1

        # Performing the Gaussian elimination as in section 4.1
        r: int = _GaussianElimination(M, 0, n - k, 0, n)
        _ = _GaussianElimination(M, r, n - k, n + r, 2 * n)

        # Get matrix sub-components, as per equation 4.3:
        # A1 = M[0:r, r : (n - k)]
        A2 = M[0:r, (n - k) : n]
        # B = M[0:r, n : (n + r)]
        C1 = M[0:r, (n + r) : (2 * n - k)]
        C2 = M[0:r, (2 * n - k) : (2 * n)]
        # D = M[r : (n - k), n : (n + r)]
        E = M[r : (n - k), (2 * n - k) : (2 * n)]

        X = np.concatenate(
            [
                np.zeros((k, r), dtype=np.int8),
                E.T,
                np.eye(k, dtype=np.int8),
                np.mod(E.T @ C1.T + C2.T, 2),
                np.zeros((k, n - k - r), np.int8),
                np.zeros((k, k), np.int8),
            ],
            axis=1,
        )

        Z = np.concatenate(
            [
                np.zeros((k, n), dtype=np.int8),
                A2.T,
                np.zeros((k, n - k - r), dtype=np.int8),
                np.eye(k, dtype=np.int8),
            ],
            axis=1,
        )

        self.n: int = n
        self.k: int = k
        self.r: int = r
        self.M: List[str] = _BuildByCode(M)
        self.X: List[str] = _BuildByCode(X)
        self.Z: List[str] = _BuildByCode(Z)
        self.syndromes_to_corrections = {}

        for qid in range(self.n):
            for op in allowed_errors:
                syndrome = tuple(
                    1 if self.M[r][qid] == 'I' or self.M[r][qid] == op else -1
                    for r in range(self.n - self.k)
                )
                self.syndromes_to_corrections[syndrome] = (op, qid)

    def encode(self, qubits: List[cirq.Qid]) -> cirq.Circuit:
        """
        Creates a circuit that encodes the qubits using the code words.

        Args:
            qubits: The list of qubits where to encode the message. This should be a vector of
            length self.n where the last self.k qubits are the un-encoded qubits.

        Returns:
            A circuit where the self.n qubits are the encoded qubits.
        """
        circuit = cirq.Circuit()

        # Equation 4.8:
        for r, x in enumerate(self.X):
            for j in range(self.r, self.n - self.k):
                if x[j] == 'X' or x[j] == 'Y':
                    circuit.append(
                        cirq.ControlledOperation([qubits[self.n - self.k + r]], cirq.X(qubits[j]))
                    )

        gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}

        for r in range(self.r):
            circuit.append(cirq.H(qubits[r]))

            if self.M[r][r] == 'Y' or self.M[r][r] == 'Z':
                circuit.append(cirq.S(qubits[r]))

            for n in range(self.n):
                if n == r:
                    continue
                if self.M[r][n] == 'I':
                    continue
                op = gate_dict[self.M[r][n]]
                circuit.append(cirq.ControlledOperation([qubits[r]], op(qubits[n])))
        # At this stage, the state vector should be equal to equations 3.17 and 3.18.

        return circuit

    def correct(self, qubits: List[cirq.Qid], ancillas: List[cirq.Qid]) -> cirq.Circuit:
        """
        Creates a correction circuit by computing the syndrom on the ancillas, and then using this
        syndrome to correct the qubits.correct

        Args:
            qubits: a vector of self.n qubits that contains (potentially corrupted) code words
            ancillas: a vector of self.n - self.k qubits that are set to zero and will contain the
                syndrome once the circuit is applied.

        Returns:
            The circuit that both computes the syndrome and uses this syndrome to correct errors.
        """
        circuit = cirq.Circuit()

        gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}

        # We set the ancillas so that measuring them directly would be the same
        # as measuring the qubits with Pauli strings. In other words, we store
        # the syndrome inside the ancillas.
        for r in range(self.n - self.k):
            for n in range(self.n):
                if self.M[r][n] == 'Z':
                    circuit.append(cirq.ControlledOperation([qubits[n]], cirq.X(ancillas[r])))
                elif self.M[r][n] == 'X':
                    circuit.append(cirq.H(qubits[n]))
                    circuit.append(cirq.ControlledOperation([qubits[n]], cirq.X(ancillas[r])))
                    circuit.append(cirq.H(qubits[n]))
                elif self.M[r][n] == 'Y':
                    circuit.append(cirq.S(qubits[n]) ** -1)
                    circuit.append(cirq.H(qubits[n]))
                    circuit.append(cirq.ControlledOperation([qubits[n]], cirq.X(ancillas[r])))
                    circuit.append(cirq.H(qubits[n]))
                    circuit.append(cirq.S(qubits[n]))

        # At this stage, the ancillas are equal to the syndrome. Now, we apply
        # the errors back to correct the code.

        for syndrome, correction in self.syndromes_to_corrections.items():
            op = gate_dict[correction[0]]
            n = correction[1]

            # We do a Boolean operation on the ancillas (i.e. syndrome).
            for r in range(self.n - self.k):
                if syndrome[r] == 1:
                    circuit.append(cirq.X(ancillas[r]))

            circuit.append(cirq.ControlledOperation(ancillas, op(qubits[n])))

            for r in range(self.n - self.k):
                if syndrome[r] == 1:
                    circuit.append(cirq.X(ancillas[r]))

        return circuit

    def decode(self, qubits: List[cirq.Qid], ancillas: List[cirq.Qid], state_vector) -> List[int]:
        """
        Computes the output of the circuit by projecting onto the \bar{Z}.

        Args:
            qubit: the qubits where the (now corrected) code words are stored.
            ancillas: the qubits where the syndrome is stored
            state_vector: a vector containing the state of the entire circuit

        Returns:
            The decoded and measured code words.
        """
        qubit_map = {qubit: i for i, qubit in enumerate(qubits + ancillas)}

        decoded = []
        for z in self.Z:
            pauli_string: cirq.PauliString = cirq.PauliString(dict(zip(qubits, z)))
            trace = pauli_string.expectation_from_state_vector(state_vector, qubit_map)
            decoded.append(round((1 - trace.real) / 2))
        return decoded

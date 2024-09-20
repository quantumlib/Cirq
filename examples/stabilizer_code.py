# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import Dict, List, Tuple

import numpy as np

import cirq

# Based on:
# Stabilizer Codes and Quantum Error Correction
# Daniel Gottesman
# https://thesis.library.caltech.edu/2900/2/THESIS.pdf
#
# However, there were some mistakes in the thesis:
# https://www2.perimeterinstitute.ca/personal/dgottesman/thesis-errata.html
#
# An alternative explanation can be found in exercises 10.3 and 10.4 of
# "Quantum Computation and Quantum Information" by Michael Nielsen and Isaac Chuang.


def _build_by_code(mat: np.ndarray) -> List[str]:
    """Transforms a matrix of Booleans into a list of Pauli strings.

    Takes into input a matrix of Boolean interpreted as row-vectors, each having dimension 2 * n.
    The matrix is converted into another matrix with as many rows, but this time the vectors
    contain the letters I, X, Y, and Z representing Pauli operators.

    Args:
        mat: The input matrix of Booleans.

    Returns:
        A list of Pauli strings.
    """
    out = []
    n = mat.shape[1] // 2
    for i in range(mat.shape[0]):
        ps = ''
        for j in range(n):
            k = 2 * mat[i, j + n] + mat[i, j]
            ps += "IXZY"[k]
        out.append(ps)
    return out


# It was considered to use scipy.linalg.lu but it seems to be only for real numbers and does
# not allow to restrict only on a section of the matrix.
def _gaussian_elimination(
    M: np.ndarray, min_row: int, max_row: int, min_col: int, max_col: int
) -> int:
    """Gaussian elimination for standard form.

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
    assert M.shape[1] % 2 == 0
    n = M.shape[1] // 2

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

        # Swap the rows:
        M[[i, i + pi]] = M[[i + pi, i]]

        # Swap the columns:
        M[:, [(j + pj), j]] = M[:, [j, (j + pj)]]

        # Since the columns in the left and right half of the matrix represent the same qubit, we
        # also need to swap the corresponding column in the other half.
        j_other_half = (j + n) % (2 * n)
        M[:, [(j_other_half + pj), j_other_half]] = M[:, [j_other_half, (j_other_half + pj)]]

        # Do the elimination.
        for k in range(i + 1, max_row):
            if M[k, j] == 1:
                M[k, :] = np.mod(M[i, :] + M[k, :], 2)

        rank += 1

    # Backward replacing to get identity
    for r in reversed(range(rank)):
        i = min_row + r
        j = min_col + r

        # Do the elimination in reverse.
        for k in range(i - 1, min_row - 1, -1):
            if M[k, j] == 1:
                M[k, :] = np.mod(M[i, :] + M[k, :], 2)

    return rank


def _transfer_to_standard_form(
    M: np.ndarray, n: int, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Puts the stabilizer matrix in its standardized form, as in section 4.1 of the thesis.

    Args:
        M: The stabilizier matrix, to be standardized.
        n: Dimension of the code words.
        k: Dimension of the message words.

    Returns:
        The standardized matrix.
        The logical Xs.
        The logical Zs.
        The rank of the matrix.
    """

    # Performing the Gaussian elimination as in section 4.1
    r: int = _gaussian_elimination(M, 0, n - k, 0, n)
    _ = _gaussian_elimination(M, r, n - k, n + r, 2 * n)

    # Get matrix sub-components, as per equation 4.3:
    A2 = M[0:r, (n - k) : n]
    C1 = M[0:r, (n + r) : (2 * n - k)]
    C2 = M[0:r, (2 * n - k) : (2 * n)]
    E = M[r : (n - k), (2 * n - k) : (2 * n)]

    X = np.concatenate(
        [
            np.zeros((k, r), dtype=np.int8),
            E.T,
            np.eye(k, dtype=np.int8),
            np.mod(E.T @ C1.T + C2.T, 2),
            np.zeros((k, n - r), np.int8),
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
    return M, X, Z, r


class StabilizerCode(object):
    def __init__(self, group_generators: List[str], correctable_errors: List[str]):
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

        M, X, Z, r = _transfer_to_standard_form(M, n, k)

        self.n: int = n
        self.k: int = k
        self.r: int = r
        self.M: List[str] = _build_by_code(M)
        self.logical_Xs: List[str] = _build_by_code(X)
        self.logical_Zs: List[str] = _build_by_code(Z)

        self.syndromes_to_corrections: Dict[Tuple[int, ...], Tuple[str, int]] = {}

        for qid in range(self.n):
            for op in correctable_errors:
                syndrome = tuple(
                    1 if self.M[r][qid] == 'I' or self.M[r][qid] == op else -1
                    for r in range(self.n - self.k)
                )
                self.syndromes_to_corrections[syndrome] = (op, qid)

    def encode(
        self, additional_qubits: List[cirq.Qid], unencoded_qubits: List[cirq.Qid]
    ) -> cirq.Circuit:
        """Creates a circuit that encodes the qubits using the code words.

        Args:
            additional_qubits: The list of self.n - self.k qubits needed to encode the qubit.
            unencoded_qubits: The list of self.k qubits that contain the original message.

        Returns:
            A circuit where the self.n qubits are the encoded qubits.
        """
        assert len(additional_qubits) == self.n - self.k
        assert len(unencoded_qubits) == self.k
        qubits = additional_qubits + unencoded_qubits

        circuit = cirq.Circuit()

        # Equation 4.8:
        # This follows the improvements of:
        # https://cs269q.stanford.edu/projects2019/stabilizer_code_report_Y.pdf
        for r, x in enumerate(self.logical_Xs):
            for j in range(self.r, self.n - self.k):
                # By constructions, the first r rows can never contain a Z, as
                # r is defined by the Gaussian elimination as the rank.
                if x[j] == 'X' or x[j] == 'Y':
                    circuit.append(
                        cirq.ControlledOperation(
                            [unencoded_qubits[r]], cirq.X(additional_qubits[j])
                        )
                    )

        gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}

        for r in range(self.r):
            circuit.append(cirq.H(qubits[r]))

            # Let's consider the first stabilizer:
            # The reason for adding S gate is Y gate we used is the complex format (i.e. to
            # make it Hermitian). It has following four cases: (ignore the phase factor)
            # (I+X@P_2...P_k)|0...0> = |0...0> + |1>|\psi>
            # (I+Y@P_2...P_k)|0...0> = |0...0> + i|1>|\psi>
            # The other forms are not possible in the standard form, by construction.

            # The first case means we need [1,1] vector and controlled gates and in the
            # second case we need [1, i] vector and controlled gates. Corresponding, it is
            # the first column of H and the first column of SH respectively.

            # For the other stabilizers, the process can be repeated, as by definition they
            # commute.

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
        """Corrects the code word.

        Creates a correction circuit by computing the syndrome on the ancillas, and then using this
        syndrome to correct the qubits.

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
        """Computes the output of the circuit by projecting onto the \bar{Z}.

        Args:
            qubits: the qubits where the (now corrected) code words are stored.
            ancillas: the qubits where the syndrome is stored
            state_vector: a vector containing the state of the entire circuit

        Returns:
            The decoded and measured code words.
        """
        qubit_map = {qubit: i for i, qubit in enumerate(qubits + ancillas)}

        decoded = []
        for z in self.logical_Zs:
            pauli_string: cirq.PauliString = cirq.PauliString(dict(zip(qubits, z)))
            trace = pauli_string.expectation_from_state_vector(state_vector, qubit_map)
            decoded.append(round((1 - trace.real) / 2))
        return decoded

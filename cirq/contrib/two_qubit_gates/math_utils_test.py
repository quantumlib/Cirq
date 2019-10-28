import numpy
import pytest
from cirq import kak_decomposition
from cirq.contrib.two_qubit_gates.math_utils import vector_kron, random_two_qubit_unitaries_and_kak_vecs

numpy.random.seed(11)


def test_vector_kron_typical_input():
    A = numpy.random.rand(5, 2, 2)
    B = numpy.random.rand(5, 3, 3)
    actual = vector_kron(A, B)
    expected = [numpy.kron(a, b) for a, b in zip(A, B)]
    numpy.testing.assert_almost_equal(actual, expected)


def _four_2Q_unitaries():
    unitaries = []

    g = numpy.random.random((4, 4)) + numpy.random.random((4, 4)) * 1j
    g += g.conj().T
    evals, evecs = numpy.linalg.eigh(g)
    unitaries.append(numpy.einsum('ab,b,cb->ac', evecs, numpy.exp(1j * evals),
                                  evecs.conj()))

    theta = 1.3
    c, s = numpy.cos(theta), numpy.sin(theta)
    unitaries.append(c * numpy.eye(4) + 1j * s * numpy.diag([1, -1, -1, 1]))
    unitaries.append(numpy.diag([1, 1, 1, 1j]))
    unitaries.append(numpy.eye(4))

    return unitaries


_unitaries, _k_vecs = random_two_qubit_unitaries_and_kak_vecs(100)




CNOT = numpy.zeros((4, 4))
CNOT[(0, 1, 2, 3), (0, 1, 3, 2)] = 1
SWAP = numpy.zeros((4, 4))
SWAP[(0, 1, 2, 3), (0, 2, 1, 3)] = 1
CZ = numpy.zeros((4, 4))
CZ[(0, 1, 2, 3), (0, 1, 2, 3)] = [1, 1, 1, -1]
ISWAP = SWAP.astype(complex)
ISWAP[(1, 2), (2, 1)] = 1j
XX = numpy.zeros((4, 4))
XX[(0, 1, 2, 3), (3, 2, 1, 0)] = 1.0

invs = [(0, 0, 1), (-1, 0, -3), (0, 0, 1), (1, 0, 3)]
cases = tuple(zip((CNOT, SWAP, CZ, CZ @ CZ), invs))


cases = [(numpy.eye(4), (0, 0, 0)),
         (SWAP, numpy.ones((3,)) * numpy.pi / 4),
         (ISWAP, [numpy.pi / 4, numpy.pi / 4, 0]),
         (ISWAP.conj(), [numpy.pi / 4, numpy.pi / 4, 0]),
         (CNOT, [numpy.pi / 4, 0, 0]),
         (CNOT @ SWAP, [numpy.pi / 4, numpy.pi / 4, 0]),
         (XX, [0, 0, 0])
         ]



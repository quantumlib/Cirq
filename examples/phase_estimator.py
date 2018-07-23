"""Creates and simulates a phase estimator circuit.

=== EXAMPLE OUTPUT ===
Estimation with 2qbits.
Actual, Estimation (Raw binary)
0.0000, 0.0000 (00)
0.1000, 0.0000 (00)
0.2000, 0.2500 (01)
0.3000, 0.2500 (01)
0.4000, 0.5000 (10)
0.5000, 0.5000 (10)
0.6000, 0.5000 (10)
0.7000, 0.7500 (11)
0.8000, 0.7500 (11)
0.9000, 0.0000 (00)
RMS Error: 0.2915

Estimation with 4qbits.
Actual, Estimation (Raw binary)
0.0000, 0.0000 (0000)
0.1000, 0.1250 (0010)
0.2000, 0.1875 (0011)
0.3000, 0.3125 (0101)
0.4000, 0.3750 (0110)
0.5000, 0.5000 (1000)
0.6000, 0.6250 (1010)
0.7000, 0.6875 (1011)
0.8000, 0.8125 (1101)
0.9000, 0.8750 (1110)
RMS Error: 0.0177

Estimation with 8qbits.
Actual, Estimation (Raw binary)
0.0000, 0.0000 (00000000)
0.1000, 0.1016 (00011010)
0.2000, 0.1992 (00110011)
0.3000, 0.3008 (01001101)
0.4000, 0.3984 (01100110)
0.5000, 0.5000 (10000000)
0.6000, 0.6016 (10011010)
0.7000, 0.6992 (10110011)
0.8000, 0.8008 (11001101)
0.9000, 0.8984 (11100110)
RMS Error: 0.0011
"""


from collections import Counter
import numpy as np

import cirq
from cirq.ops import ControlledGate
from cirq.ops import SingleQubitMatrixGate


def mode(lst):

    """Return the most common element from a list.
    """

    data = Counter(lst)
    return data.most_common(1)[0][0]


def phase_op(phi):

    """An example unitary 1-qubit operation U with an eigen vector |0>
    and an eigen value exp(2*Pi*i*phi)
    """

    gate = SingleQubitMatrixGate(
        matrix=np.array([[np.exp(2*np.pi*1.0j*phi), 0], [0, 1]]))
    return gate


class QftInverse(cirq.Gate, cirq.CompositeGate):

    """Quantum gate for the inverse Quantum Fourier Transformation
    """

    def cr_inverse(self, k):

        """A controlled unitary gate R_k that applies the following phase shift:
        R_k|0> = |0>
        R_k|1> = exp(-2 Pi i / 2^k)|1>
        """

        r_k = SingleQubitMatrixGate(
            matrix=np.array([[1, 0], [0, np.exp(-2*np.pi*1.0j/2**k)]]))
        return ControlledGate(r_k)


    def default_decompose(self, qubits):

        """A quantum circuit with the following recursive structure.
        ---H----R_2---R_3---R_4--------------------------------------
                 |     |     |
        ---------@-----|-----|-----H----R_2---R_3--------------------
                       |     |           |     |                       = QFT_inv
        ---------------@-----|-----------@-----|-----H----R_2--------
                             |                 |           |
        ---------------------@-----------------@-----------@-----H---

        where R_k is an inverse phase shift U as below:
        U|0> = |0>
        U|1> = exp(-2 Pi i / 2^k)|1>

        The number of qubits can be arbitrary.
        """

        qubits = list(qubits)
        while len(qubits) > 0:
            q_head = qubits.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qubits):
                yield self.cr_inverse(i+2)(qubit, q_head)


def run_estimate(phi, qnum, repeats):

    """Construct the following phase estimator circuit and execute simulations.
                                     ---------
    ---H---------------------@------|         |---[m3]---
                             |      |         |
    ---H---------------@-----|------|         |---[m2]---
                       |     |      | QFT_inv |
    ---H---------@-----|-----|------|         |---[m1]---
                 |     |     |      |         |
    ---H---@-----|-----|-----|------|         |---[m0]---
           |     |     |     |       ---------
    -------U----U^2---U^4---U^8--------------------------

    The measurement results are translated to the estimated phase with the
    formula: phi = m0*(1/2) + m1*(1/2)^2 + m2*(1/2)^3 + m3*(1/2)^4 + ...
    """

    qubits = [None] * qnum
    for i in range(len(qubits)):
        qubits[i] = cirq.GridQubit(0, i)
    ansilla = cirq.GridQubit(0, len(qubits))

    ops = [cirq.H(q) for q in qubits]
    ops += [ControlledGate(phase_op((2**i)*phi))(qubits[qnum-i-1], ansilla)
            for i in range(qnum)]
    ops.append(QftInverse()(*qubits))
    ops += [cirq.measure(qubits[len(qubits)-i-1], key='m{}'.format(i))
            for i in range(qnum)]
    circuit = cirq.Circuit.from_ops(*ops)
    simulator = cirq.google.XmonSimulator()
    result = simulator.run(circuit, repetitions=repeats)
    return result


def experiment(qnum, repeats=100):

    """Execute the phase estimator cirquit with multiple settings and
    show results.
    """

    print('Estimation with {}qbits.'.format(qnum))
    print('Actual, Estimation (Raw binary)')
    errors = []
    for phi in np.arange(0, 1, 0.1):
        result = run_estimate(phi, qnum, repeats)
        measurements_bool = zip(*[result.measurements['m{}'.format(i)].flatten()
                                  for i in range(qnum)])
        measurements_bin = [('{:d}'*qnum).format(*m) for m in measurements_bool]
        estimate_bin = mode(measurements_bin)
        estimate = (sum([float(s)*0.5**(order+1)
                         for order, s in enumerate(estimate_bin)]))
        print('{:0.4f}, {:0.4f} ({})'.format(phi, estimate, estimate_bin))
        errors.append((phi-estimate)**2)
    print('RMS Error: {:0.4f}\n'.format(np.sqrt(sum(errors)/len(errors))))


def main():
    for qnum in [2, 4, 8]:
        experiment(qnum)


if __name__ == '__main__':
    main()

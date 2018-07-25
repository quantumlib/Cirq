"""Creates and simulates a phase estimator circuit.

=== EXAMPLE OUTPUT ===
Estimation with 2qubits.
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

Estimation with 4qubits.
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

Estimation with 8qubits.
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


import numpy as np
import cirq


class QftInverse(cirq.Gate, cirq.CompositeGate):
    """Quantum gate for the inverse Quantum Fourier Transformation
    """

    def default_decompose(self, qubits):
        """A quantum circuit (QFT_inf) with the following structure.

        ---H--@-------@--------@----------------------------------------------
              |       |        |
        ------@^-0.5--+--------+---------H--@-------@-------------------------
                      |        |            |       |
        --------------@^-0.25--+------------@^-0.5--+---------H--@------------
                               |                    |            |
        -----------------------@^-0.125-------------@^-0.25------@^-0.5---H---

        The number of qubits can be arbitrary.
        """

        qubits = list(qubits)
        while len(qubits) > 0:
            q_head = qubits.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qubits):
                yield (cirq.CZ**(-1/2.0**(i+1)))(qubit, q_head)


def run_estimate(unknown_gate, qnum, repeats):
    """Construct the following phase estimator circuit and execute simulations.

                                     ---------
    ---H---------------------@------|         |---M--- [m4]:lowest bit
                             |      |         |
    ---H---------------@-----|------|         |---M--- [m3]
                       |     |      | QFT_inv |
    ---H---------@-----|-----|------|         |---M--- [m2]
                 |     |     |      |         |
    ---H---@-----|-----|-----|------|         |---M--- [m1]:highest bit
           |     |     |     |       ---------
    -------U-----U^2---U^4---U^8----------------------

    The measurement results M=[m1, m2,...] are translated to the estimated
    phase with the following formula:
    phi = m1*(1/2) + m2*(1/2)^2 + m3*(1/2)^3 + ...
    """

    qubits = [None] * qnum
    for i in range(len(qubits)):
        qubits[i] = cirq.GridQubit(0, i)
    ancilla = cirq.GridQubit(0, len(qubits))

    circuit = cirq.Circuit.from_ops(
        cirq.H.on_each(qubits),
        [cirq.ControlledGate(unknown_gate**(2**i)).on(qubits[qnum-i-1], ancilla)
         for i in range(qnum)],
        QftInverse()(*qubits),
        cirq.measure(*qubits, key='phase'))
    simulator = cirq.google.XmonSimulator()
    result = simulator.run(circuit, repetitions=repeats)
    return result


def experiment(qnum, repeats=100):
    """Execute the phase estimator cirquit with multiple settings and
    show results.
    """

    def example_gate(phi):
        """An example unitary 1-qubit gate U with an eigen vector |0> and an
        eigen value exp(2*Pi*i*phi)
        """

        gate = cirq.SingleQubitMatrixGate(
            matrix=np.array([[np.exp(2*np.pi*1.0j*phi), 0], [0, 1]]))
        return gate

    print('Estimation with {}qubits.'.format(qnum))
    print('Actual, Estimation (Raw binary)')
    errors = []
    fold_func = lambda ms: ''.join(np.flip(ms, 0).astype(int).astype(str))
    for phi in np.arange(0, 1, 0.1):
        result = run_estimate(example_gate(phi), qnum, repeats)
        hist = result.histogram(key='phase', fold_func=fold_func)
        estimate_bin = hist.most_common(1)[0][0]
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

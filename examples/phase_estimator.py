"""Creates and simulates a phase estimator circuit.

=== EXAMPLE OUTPUT ===
Testing with 8 qubits.
target=0.0000, estimate=0.0000=0/256
target=0.1000, estimate=0.1016=26/256
target=0.2000, estimate=0.1992=51/256
target=0.3000, estimate=0.3008=77/256
target=0.4000, estimate=0.3984=102/256
target=0.5000, estimate=0.5000=128/256
target=0.6000, estimate=0.6016=154/256
target=0.7000, estimate=0.6992=179/256
target=0.8000, estimate=0.8008=205/256
target=0.9000, estimate=0.8984=230/256
RMS Error: 0.0011
"""


import numpy as np
import cirq


def run_estimate(unknown_gate, qnum, repetitions):
    """Construct the following phase estimator circuit and execute simulations.

                                     ---------
    ---H---------------------@------|         |---M--- [m4]:lowest bit
                             |      |         |
    ---H---------------@-----+------|         |---M--- [m3]
                       |     |      | QFT_inv |
    ---H---------@-----+-----+------|         |---M--- [m2]
                 |     |     |      |         |
    ---H---@-----+-----+-----+------|         |---M--- [m1]:highest bit
           |     |     |     |       ---------
    -------U-----U^2---U^4---U^8----------------------

    The measurement results M=[m1, m2,...] are translated to the estimated
    phase with the following formula:
    phi = m1*(1/2) + m2*(1/2)^2 + m3*(1/2)^3 + ...
    """

    ancilla = cirq.LineQubit(-1)
    qubits = cirq.LineQubit.range(qnum)

    oracle_raised_to_power = [
        unknown_gate.on(ancilla).controlled_by(qubits[i]) ** (2 ** i) for i in range(qnum)
    ]
    circuit = cirq.Circuit(
        cirq.H.on_each(*qubits),
        oracle_raised_to_power,
        cirq.qft(*qubits, without_reverse=True) ** -1,
        cirq.measure(*qubits, key='phase'),
    )

    return cirq.sample(circuit, repetitions=repetitions)


def experiment(qnum, repetitions=100):
    """Execute the phase estimator circuit with multiple settings and
    show results.
    """

    def example_gate(phi):
        """An example unitary 1-qubit gate U with an eigen vector |0> and an
        eigen value exp(2*Pi*i*phi)
        """

        gate = cirq.MatrixGate(matrix=np.array([[np.exp(2 * np.pi * 1.0j * phi), 0], [0, 1]]))
        return gate

    print(f'Testing with {qnum} qubits.')
    errors = []
    for target in np.arange(0, 1, 0.1):
        result = run_estimate(example_gate(target), qnum, repetitions)
        mode = result.data['phase'].mode()[0]
        guess = mode / 2 ** qnum
        print(f'target={target:0.4f}, estimate={guess:0.4f}={mode}/{2**qnum}')
        errors.append((target - guess) ** 2)
    rms = np.sqrt(sum(errors) / len(errors))
    print(f'RMS Error: {rms:0.4f}\n')


def main(qnums=(2, 4, 8), repetitions=100):
    for qnum in qnums:
        experiment(qnum, repetitions=repetitions)


if __name__ == '__main__':
    main()

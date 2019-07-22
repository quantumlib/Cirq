"""
Creates and simulate a noisy circuit using cirq.ConstantQubitNoiseModel class.
"""
import cirq


def noisyCircuitDemo(amplitude_damp):
    """Demonstrates a noisy circuit simulation.
    """
    q = cirq.NamedQubit('q')
    noisy_circuit = cirq.Circuit.from_ops(
        cirq.measure(q, key='initial_state'),
        cirq.X(q),
        cirq.measure(q, key='after_not_gate'),
    )
    results = cirq.sample(program=noisy_circuit,
                          noise=cirq.ConstantQubitNoiseModel(
                              cirq.amplitude_damp(amplitude_damp)),
                          repetitions=100)
    print("Noise model: ConstantQubitNoiseModel with",
          cirq.amplitude_damp(amplitude_damp))
    print('Sampling of initial state of qubit "q":')
    print(results.histogram(key='initial_state'))
    print('Sampling of qubit "q" after application of X gate:')
    print(results.histogram(key='after_not_gate'))


def main():
    print("Iteration 1:")
    noisyCircuitDemo(0)
    print()
    print("Iteration 2:")
    noisyCircuitDemo(0.4)
    print()
    print("Iteration 3:")
    noisyCircuitDemo(0.5)
    print()
    print("Iteration 4:")
    noisyCircuitDemo(1.0)


if __name__ == '__main__':
    main()

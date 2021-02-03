"""
Creates and simulate a noisy circuit using cirq.ConstantQubitNoiseModel class.
"""
import cirq


def noisy_circuit_demo(amplitude_damp):
    """Demonstrates a noisy circuit simulation."""
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(
        cirq.measure(q, key='initial_state'),
        cirq.X(q),
        cirq.measure(q, key='after_not_gate'),
    )
    results = cirq.sample(
        program=circuit,
        noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(amplitude_damp)),
        repetitions=100,
    )
    print(
        "ConstantQubitNoiseModel with amplitude damping of rate",
        cirq.amplitude_damp(amplitude_damp),
    )
    print('Sampling of initial state of qubit "q":')
    print(results.histogram(key='initial_state'))
    print('Sampling of qubit "q" after application of X gate:')
    print(results.histogram(key='after_not_gate'))


def main():
    amp_damp_rates = [0, 0.4, 0.5, 1.0]
    for amp_damp_rate in amp_damp_rates:
        noisy_circuit_demo(amp_damp_rate)
        print()


if __name__ == '__main__':
    main()

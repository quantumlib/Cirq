import cirq


def test_should_only_one():
    simulator = cirq.Simulator()
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q, key='x'))
    simulator.simulate(circuit)
    simulator.simulate(circuit)

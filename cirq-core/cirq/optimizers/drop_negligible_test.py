import cirq
from cirq.ops.tags import NoCompileTag


def assert_optimizes(optimizer, initial_circuit: cirq.Circuit, expected_circuit: cirq.Circuit):
    circuit = cirq.Circuit(initial_circuit)
    optimizer.optimize_circuit(circuit)
    assert circuit == expected_circuit


def test_leaves_big():
    drop = cirq.DropNegligible(0.001)
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit([cirq.Moment([cirq.Z(a) ** 0.1])])
    circuit2 = cirq.Circuit([cirq.Moment([(cirq.Z(a) ** 0.1).with_tags(NoCompileTag)])])

    assert_optimizes(optimizer=drop, initial_circuit=circuit, expected_circuit=circuit)
    assert_optimizes(optimizer=drop, initial_circuit=circuit2, expected_circuit=circuit2)


def test_clears_small():
    drop = cirq.DropNegligible(0.001)
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit([cirq.Moment([cirq.Z(a) ** 0.000001])])
    circuit2 = cirq.Circuit([cirq.Moment([(cirq.Z(a) ** 0.000001).with_tags(NoCompileTag)])])

    assert_optimizes(
        optimizer=drop, initial_circuit=circuit, expected_circuit=cirq.Circuit([cirq.Moment()])
    )

    assert_optimizes(
        optimizer=drop, initial_circuit=circuit2, expected_circuit=circuit2
    )


def test_clears_known_empties_even_at_zero_tolerance():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.Z(a) ** 0, cirq.Y(a) ** 0.0000001, cirq.X(a) ** -0.0000001, cirq.CZ(a, b) ** 0
    )

    circuit2 = cirq.Circuit(
        (cirq.Z(a) ** 0).with_tags(NoCompileTag), cirq.Y(a) ** 0.0000001, cirq.X(a) ** -0.0000001, cirq.CZ(a, b) ** 0
    )
    assert_optimizes(
        optimizer=cirq.DropNegligible(tolerance=0.001),
        initial_circuit=circuit,
        expected_circuit=cirq.Circuit([cirq.Moment()] * 4),
    )
    assert_optimizes(
        optimizer=cirq.DropNegligible(tolerance=0),
        initial_circuit=circuit,
        expected_circuit=cirq.Circuit(
            [
                cirq.Moment(),
                cirq.Moment([cirq.Y(a) ** 0.0000001]),
                cirq.Moment([cirq.X(a) ** -0.0000001]),
                cirq.Moment(),
            ]
        ),
    )
    assert_optimizes(
        optimizer=cirq.DropNegligible(tolerance=0),
        initial_circuit=circuit2,
        expected_circuit=cirq.Circuit(
            [
                cirq.Moment([(cirq.Z(a) ** 0).with_tags(NoCompileTag)]),
                cirq.Moment([cirq.Y(a) ** 0.0000001]),
                cirq.Moment([cirq.X(a) ** -0.0000001]),
                cirq.Moment(),
            ]
        ),
    )

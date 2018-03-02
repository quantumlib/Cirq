from cirq.examples import generate_supremacy_circuit
from cirq.google import Foxtail, programs
from cirq.schedules import moment_by_moment_schedule


def test_protobuf_roundtrip():
    device = Foxtail
    circuit = generate_supremacy_circuit(device, cz_depth=6)
    s1 = moment_by_moment_schedule(device, circuit)

    protos = list(programs.schedule_to_proto(s1))
    s2 = programs.schedule_from_proto(device, protos)

    s1ops, s2ops = s1.scheduled_operations, s2.scheduled_operations
    assert len(s1ops) == len(s2ops)
    for s1op, s2op in zip(s1ops, s2ops):
        assert s1op == s2op

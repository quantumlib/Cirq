from cirq.examples import generate_supremacy_circuit
from cirq.google import Foxtail, programs
from cirq.schedules import moment_by_moment_schedule


def test_protobuf_roundtrip():
    device = Foxtail
    circuit = generate_supremacy_circuit(device, cz_depth=6)
    s1 = moment_by_moment_schedule(device, circuit)

    protos = list(programs.schedule_to_proto(s1))
    s2 = programs.schedule_from_proto(device, protos)

    assert s2 == s1

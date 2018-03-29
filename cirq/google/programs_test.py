import numpy as np

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


def make_bytes(s: str) -> bytes:
    """Helper function to convert a string of digits into packed bytes.

    Ignores any characters other than 0 and 1, in particular whitespace. The
    bits are packed in little-endian order within each byte.
    """
    buf = []
    byte = 0
    idx = 0
    for c in s:
        if c == '0':
            pass
        elif c == '1':
            byte |= 1 << idx
        else:
            continue
        idx += 1
        if idx == 8:
            buf.append(byte)
            byte = 0
            idx = 0
    if idx:
        buf.append(byte)
    return bytearray(buf)


def test_unpack_results():
    data = make_bytes("""
        000 00
        001 01
        010 10
        011 11
        100 00
        101 01
        110 10
    """)
    assert len(data) == 5  # 35 data bits + 5 padding bits
    results = programs.unpack_results(data, 7, [('a', 3), ('b', 2)])
    assert 'a' in results
    assert results['a'].shape == (7, 3)
    np.testing.assert_array_equal(
        results['a'],
        [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [1, 0, 0],
         [1, 0, 1],
         [1, 1, 0],])

    assert 'b' in results
    assert results['b'].shape == (7, 2)
    np.testing.assert_array_equal(
        results['b'],
        [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1],
         [0, 0],
         [0, 1],
         [1, 0],])

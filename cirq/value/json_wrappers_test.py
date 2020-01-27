import io

import numpy as np

import cirq
import cirq.protocols


def test_bits_roundtrip():
    bitstring = np.asarray([0, 1, 0, 1, 1, 1, 1, 0, 0, 1])
    b = cirq.BitArray(bitstring)

    buffer = io.StringIO()
    cirq.to_json(b, buffer)

    buffer.seek(0)
    text = buffer.read()
    assert text == """{
  "cirq_type": "BitArray",
  "shape": [
    10
  ],
  "packedbits": "5e40"
}"""

    buffer.seek(0)
    b2 = cirq.read_json(buffer)
    assert b == b2


def test_bits_roundtrip_big():
    bits = np.random.choice([0, 1], size=(30_000, 53))
    b = cirq.BitArray(bits)
    buffer = io.StringIO()
    cirq.to_json(b, buffer)
    buffer.seek(0)
    b2 = cirq.read_json(buffer)
    assert b == b2

    bits = np.random.choice([0, 1], size=(3000, 11, 53))
    b = cirq.BitArray(bits)
    buffer = io.StringIO()
    cirq.to_json(b, buffer)
    buffer.seek(0)
    b2 = cirq.read_json(buffer)
    assert b == b2


def test_bitstrings_roundtrip_big():
    bitstrings = np.random.choice([0, 1], size=(30_000, 53))
    ba = cirq.BitArray(bitstrings)

    buffer = io.StringIO()
    cirq.to_json(ba, buffer)
    buffer.seek(0)
    ba2 = cirq.read_json(buffer)
    assert ba == ba2


def test_numpy_roundtrip(tmpdir):
    re = np.random.uniform(0, 1, 100)
    im = np.random.uniform(0, 1, 100)
    a = re + 1.j * im
    a = np.reshape(a, (10, 10))
    ba = cirq.NumpyArray(a)

    fn = f'{tmpdir}/hello.json'
    cirq.to_json(ba, fn)
    ba2 = cirq.read_json(fn)

    assert ba == ba2

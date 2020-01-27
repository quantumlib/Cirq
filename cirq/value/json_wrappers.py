import io
from typing import List

import numpy as np


class NumpyArray:
    """Support for compact serialization of a numpy array.

    Instead of transforming data to a list-of-lists, this hex-encodes
    a binary representation of the numpy array.
    """

    def __init__(self, a: np.ndarray):
        self.a = a

    def _json_dict_(self):
        buffer = io.BytesIO()
        np.save(buffer, self.a, allow_pickle=False)
        buffer.seek(0)
        d = {
            'cirq_type': self.__class__.__name__,
            'npy': buffer.read().hex(),
        }
        buffer.close()
        return d

    @classmethod
    def _from_json_dict_(cls, npy: str, **kwargs):
        buffer = io.BytesIO()
        buffer.write(bytes.fromhex(npy))
        buffer.seek(0)
        a = np.load(buffer, allow_pickle=False)
        buffer.close()
        return cls(a)

    def __str__(self):
        return f'cirq.NumpyArray({self.a})'

    def __repr__(self):
        return f'cirq.NumpyArray({repr(self.a)})'

    def __eq__(self, other):
        return np.array_equal(self.a, other.a)


class BitArray:
    """A serializable wrapper for arrays specifically of bits.

    This is very similar to ``NumpyArray``, except it first "packs"
    bits into uint8's, potentially saving a factor of eight in storage
    size. The resulting binary buffer is still hex encoded into
    a JSON string.
    """

    def __init__(self, bits: np.ndarray):
        self.bits = bits

    def _json_dict_(self):
        packed_bits = np.packbits(self.bits)
        assert packed_bits.dtype == np.uint8, packed_bits.dtype
        return {
            'cirq_type': self.__class__.__name__,
            'shape': self.bits.shape,
            'packedbits': packed_bits.tobytes().hex(),
        }

    @classmethod
    def _from_json_dict_(cls, shape: List[int], packedbits: str, **kwargs):
        # Hex -> bytes -> packed array -> padded array -> final array
        bits_bytes = bytes.fromhex(packedbits)
        bits = np.frombuffer(bits_bytes, dtype=np.uint8)
        bits = np.unpackbits(bits)
        bits = bits[:np.prod(shape)].reshape(shape)
        return cls(bits)

    def __str__(self):
        return f'cirq.BitArray({self.bits})'

    def __repr__(self):
        return f'cirq.BitArray({repr(self.bits)})'

    def __eq__(self, other):
        return np.array_equal(self.bits, other.bits)

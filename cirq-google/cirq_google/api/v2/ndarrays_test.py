# Copyright 2025 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import math
from collections.abc import Sequence

import numpy as np
import pytest

import cirq_google.api.v2.ndarrays as ndarrays
import cirq_google.api.v2.ndarrays_pb2 as ndarrays_pb2


@pytest.mark.parametrize("dtype", ["f8", "<f8", ">f8", "f4", "<f4", ">f4"])
@pytest.mark.parametrize("values", [[], [1, 2, 3, 4], [[1, 2], [3, 4]]])
def test_float_conversions(dtype, values):
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_float64_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_float64_array(msg)
    assert np.all(array == deserialized)
    assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("dtype", ["f4", "<f4", ">f4", "f2"])
@pytest.mark.parametrize("values", [[], [1.0, 2.0, -3.0, -4.0], [[1.0, -2.0], [3.0, -4.0]]])
def test_float32_conversions(dtype: str, values: Sequence[float]) -> None:
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_float32_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_float32_array(msg)
    assert np.all(array == deserialized)
    assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("dtype", ["f2", "<f2", "f2"])
@pytest.mark.parametrize("values", [[], [1.0, -2.0, 3.0, -4.0], [[1.0, -2.0], [-3.0, 4.0]]])
def test_float16_conversions(dtype: str, values: Sequence[float]) -> None:
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_float16_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_float16_array(msg)
    assert np.all(array == deserialized)
    assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("dtype", ["f8", "u4", "i8"])
def test_incompatible_float32_dtypes(dtype: str) -> None:
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_float32_array(array)


@pytest.mark.parametrize("dtype", ["f8", "f4", "c8", "i4", "u4", "i8", "u8"])
def test_incompatible_float16_dtypes(dtype: str) -> None:
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_float16_array(array)


@pytest.mark.parametrize("dtype", ["i8", "<i8", ">i8", "i4", "<i4", ">i4", "i2", "i1"])
@pytest.mark.parametrize("values", [[], [1, 2, -3, -4], [[1, -2], [3, -4]]])
def test_int64_conversions(dtype, values):
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_int64_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_int64_array(msg)
    assert np.all(array == deserialized)
    if array.dtype.itemsize > 1:
        assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("dtype", ["i4", "<i4", ">i4", "i2", "i1"])
@pytest.mark.parametrize("values", [[], [1, 2, -3, -4], [[1, -2], [3, -4]]])
def test_int32_conversions(dtype, values):
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_int32_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_int32_array(msg)
    assert np.all(array == deserialized)
    if array.dtype.itemsize > 1:
        assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("dtype", ["i2", "<i2", "i2", "i1"])
@pytest.mark.parametrize("values", [[], [1, -2, 3, -4], [[1, -2], [-3, 4]]])
def test_int16_conversions(dtype, values):
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_int16_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_int16_array(msg)
    assert np.all(array == deserialized)
    if array.dtype.itemsize > 1:
        assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("values", [[], [1, -2, 3, -4], [[1, -2], [-3, 4]]])
def test_int8_conversions(values):
    array = np.array(values, dtype="i1")
    msg = ndarrays.to_int8_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_int8_array(msg)
    assert np.all(array == deserialized)


@pytest.mark.parametrize("values", [[], [1, 2, 3, 4], [[1, 2], [3, 4]]])
def test_uint8_conversions(values):
    array = np.array(values, dtype="u1")
    msg = ndarrays.to_uint8_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_uint8_array(msg)
    assert np.all(array == deserialized)


@pytest.mark.parametrize("dtype", [np.uint8, bool])
@pytest.mark.parametrize("shape", [(0,), (16,), (17, 13), (5, 7, 9, 12)])
def test_bitarray_conversions(dtype, shape: tuple[int, ...]):
    # Create a (deterministically) random bit array
    np.random.seed(0xABCD0123)
    bits = np.random.choice(a=[0, 1], size=shape)

    array = np.array(bits, dtype=dtype)
    msg = ndarrays.to_bitarray(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_bitarray(msg)
    assert np.all(array == deserialized)

    # Ensure that we're correctly storing bit-packed bytes.
    tot_bits = functools.reduce(lambda x, y: x * y, shape)
    expected_byte_len = int(math.ceil(tot_bits / 8))
    assert len(msg.flat_bytes) == expected_byte_len


def test_bitarray_validation():
    # this should be ok:
    data = np.array([0, 1, 0, 1], dtype=np.uint8)
    ndarrays.to_bitarray(data)

    # now a bad array
    data = np.array([0, 1, 0, 2], dtype=np.uint8)
    with pytest.raises(ValueError, match="Cannot convert array"):
        ndarrays.to_bitarray(data)


@pytest.mark.parametrize("dtype", ["u4", "u8"])
def test_incompatible_int64_dtypes(dtype):
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_int64_array(array)


@pytest.mark.parametrize("dtype", ["u4", "i8"])
def test_incompatible_int32_dtypes(dtype):
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_int32_array(array)


@pytest.mark.parametrize("dtype", ["i4", "u4", "i8", "u8"])
def test_incompatible_int16_dtypes(dtype):
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_int16_array(array)


@pytest.mark.parametrize("dtype", ["i1", "u2", "i2"])
def test_incompatible_uint8_dtypes(dtype):
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_uint8_array(array)


@pytest.mark.parametrize("dtype", ["c16", "<c16", ">c16", "c8"])
@pytest.mark.parametrize(
    "values", [[], [1 + 2j, 3 + 4j, 5 + 6j], [[1 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], [[1, 2], [3, 4]]]
)
def test_complex128_conversions(dtype, values):
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_complex128_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_complex128_array(msg)
    assert np.all(array == deserialized)
    assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("dtype", ["c8", "<c8", ">c8"])
@pytest.mark.parametrize(
    "values", [[], [1 + 2j, 3 + 4j, 5 + 6j], [[1 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], [[1, 2], [3, 4]]]
)
def test_complex64_conversions(dtype, values):
    array = np.array(values, dtype=dtype)
    msg = ndarrays.to_complex64_array(array)
    assert tuple(msg.shape) == array.shape
    deserialized = ndarrays.from_complex64_array(msg)
    assert np.all(array == deserialized)
    assert array.dtype.byteorder == deserialized.dtype.byteorder


@pytest.mark.parametrize("dtype", ["i2", "f4"])
def test_incompatible_complex128(dtype):
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_complex128_array(array)


@pytest.mark.parametrize("dtype", ["i2", "f4", "c16", "<c16", ">c16"])
def test_incompatible_complex64(dtype):
    array = np.array([1, 2, 3, 4], dtype=dtype)
    with pytest.raises(ValueError):
        ndarrays.to_complex64_array(array)


def test_overwrite_works():
    array1 = np.array([1, 2, 3, 4], dtype="i4")
    array2 = np.array([5, 6], dtype="i4")
    msg = ndarrays_pb2.Int32Array()
    ndarrays.to_int32_array(array1, msg)
    ndarrays.to_int32_array(array2, msg)
    assert tuple(msg.shape) == array2.shape
    assert np.all(array2 == ndarrays.from_int32_array(msg))


def test_empty_messages():
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_int8_array(ndarrays_pb2.Int8Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_int16_array(ndarrays_pb2.Int16Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_int32_array(ndarrays_pb2.Int32Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_float16_array(ndarrays_pb2.Float16Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_float32_array(ndarrays_pb2.Float32Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_float64_array(ndarrays_pb2.Float64Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_complex64_array(ndarrays_pb2.Complex64Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_complex128_array(ndarrays_pb2.Complex128Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_uint8_array(ndarrays_pb2.UInt8Array())
    with pytest.raises(ValueError, match=r"Cannot convert unset"):
        ndarrays.from_bitarray(ndarrays_pb2.BitArray())

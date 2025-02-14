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
"""Utilities for converting to/from ndarray protos."""

from __future__ import annotations

import functools
import sys
from typing import cast, TypeVar

import numpy as np
import numpy.typing as npt

from cirq_google.api.v2 import ndarrays_pb2

NATIVE_BO = ndarrays_pb2.LITTLE_ENDIAN if sys.byteorder == "little" else ndarrays_pb2.BIG_ENDIAN

TIntMsg = TypeVar(
    "TIntMsg",
    ndarrays_pb2.Int8Array,
    ndarrays_pb2.Int16Array,
    ndarrays_pb2.Int32Array,
    ndarrays_pb2.Int64Array,
)
TUIntMsg = ndarrays_pb2.UInt8Array
TFloatMsg = TypeVar(
    "TFloatMsg", ndarrays_pb2.Float16Array, ndarrays_pb2.Float32Array, ndarrays_pb2.Float64Array
)
TComplexMsg = TypeVar("TComplexMsg", ndarrays_pb2.Complex128Array, ndarrays_pb2.Complex64Array)


def _to_dtype(
    dtype: npt.DTypeLike, dtype_base: npt.DTypeLike
) -> tuple[ndarrays_pb2.Endianness.V, np.dtype]:
    dtype = np.dtype(dtype)
    dtype_base = np.dtype(dtype_base)
    if dtype.kind != dtype_base.kind or dtype.itemsize > dtype_base.itemsize:
        raise ValueError(f"cannot serialize array of type {dtype} as {dtype_base}")
    match dtype.byteorder:
        case "=":
            return NATIVE_BO, dtype_base.newbyteorder("=")
        case "<":
            return ndarrays_pb2.LITTLE_ENDIAN, dtype_base.newbyteorder("<")  # pragma: no cover
        case ">":
            return ndarrays_pb2.BIG_ENDIAN, dtype_base.newbyteorder(">")
        case "|":
            # For single-byte data, default to little-endian
            return ndarrays_pb2.LITTLE_ENDIAN, dtype_base.newbyteorder("<")
        case byteorder:  # pragma: no cover
            raise ValueError(f"Unsupported byte order: {byteorder}")  # pragma: no cover


def _from_dtype(endianness: ndarrays_pb2.Endianness, dtype_base: npt.DTypeLike) -> np.dtype:
    dtype_base = np.dtype(dtype_base)
    if endianness == ndarrays_pb2.LITTLE_ENDIAN:
        return np.dtype(f"<{dtype_base.kind}{dtype_base.itemsize}")
    else:
        return np.dtype(f">{dtype_base.kind}{dtype_base.itemsize}")


def to_float64_array(
    array: np.ndarray, out: ndarrays_pb2.Float64Array | None = None
) -> ndarrays_pb2.Float64Array:
    """Populate a new or provided Float64Array message with given numpy array.

    Supports any float type taking <=8 bytes. If given e.g. a f4/32-bit float,
    this will recast as a 64-bit float and serialize.

    Args:
        array: Array to add.
        out: Message to add to. If None, create new message.
    """
    if out is None:
        out = ndarrays_pb2.Float64Array()
    endianness, dtype = _to_dtype(array.dtype, "f8")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_float64_array(msg: ndarrays_pb2.Float64Array) -> np.ndarray:
    """Convert a Float64Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with 64-bit float type.
    """
    return _from_float_array(msg, "f8")


def to_float32_array(
    array: np.ndarray, out: ndarrays_pb2.Float32Array | None = None
) -> ndarrays_pb2.Float32Array:
    """Populate a new or provided Float32Array message with given numpy array.

    Supports any float type taking <=4 bytes. If given e.g. a f2/16-bit float,
    this will recast as a 32-bit float and serialize.

    Args:
        array: Array to add.
        out: Message to add to. If None, create new message.
    """
    if out is None:
        out = ndarrays_pb2.Float32Array()
    endianness, dtype = _to_dtype(array.dtype, "f4")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_float32_array(msg: ndarrays_pb2.Float32Array) -> np.ndarray:
    """Convert a Float32Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with 32-bit float type.
    """
    return _from_float_array(msg, "f4")


def to_float16_array(
    array: np.ndarray, out: ndarrays_pb2.Float16Array | None = None
) -> ndarrays_pb2.Float16Array:
    """Populate a new or provided Float16Array message with given numpy array.

    Args:
        array: Array to add.
        out: Message to add to. If None, create new message.
    """
    if out is None:
        out = ndarrays_pb2.Float16Array()
    endianness, dtype = _to_dtype(array.dtype, "f2")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_float16_array(msg: ndarrays_pb2.Float16Array) -> np.ndarray:
    """Convert a Float16Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with 16-bit float type.
    """
    return _from_float_array(msg, "f2")


def _from_float_array(msg: TFloatMsg, dtype_base: npt.DTypeLike) -> np.ndarray:
    if not msg.shape:
        raise ValueError(f"Cannot convert unset/empty {type(msg)} message to a numpy array.")

    array_format = msg.WhichOneof("data")
    if array_format != "flat_bytes":
        raise ValueError(
            "Error deserializing ndarray proto: only flat_bytes format "
            f"supported; given {array_format}"
        )  # pragma: no cover

    dtype = _from_dtype(cast(ndarrays_pb2.Endianness, msg.endianness), dtype_base)
    flat = np.frombuffer(msg.flat_bytes, dtype=dtype)
    return np.reshape(flat, msg.shape)


def to_int64_array(
    array: np.ndarray, out: ndarrays_pb2.Int64Array | None = None
) -> ndarrays_pb2.Int64Array:
    """Populate a new or provided Int64Array message with given numpy array.

    Supports any int type taking <=8 bytes. If given e.g. an 8-bit or 16-bit
    integer, it will recast (sign extend) as a 64-bit integer and serialize.

    Note that this only accepts signed integers.

    Args:
        array: Array to add.
        out: Message to add to.
    """
    if out is None:
        out = ndarrays_pb2.Int64Array()
    endianness, dtype = _to_dtype(array.dtype, "i8")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_int64_array(msg: ndarrays_pb2.Int64Array) -> np.ndarray:
    """Convert a Int64Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with signed integer type.
    """
    return _from_int_array(msg, "i8")


def to_int32_array(
    array: np.ndarray, out: ndarrays_pb2.Int32Array | None = None
) -> ndarrays_pb2.Int32Array:
    """Populate a new or provided Int32Array message with given numpy array.

    Supports any int type taking <=4 bytes. If given e.g. an 8-bit or 16-bit
    integer, it will recast (sign extend) as a 32-bit integer and serialize.

    Note that this only accepts signed integers.

    Args:
        array: Array to add.
        out: Message to add to.
    """
    if out is None:
        out = ndarrays_pb2.Int32Array()
    endianness, dtype = _to_dtype(array.dtype, "i4")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_int32_array(msg: ndarrays_pb2.Int32Array) -> np.ndarray:
    """Convert a Int32Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with signed integer type.
    """
    return _from_int_array(msg, "i4")


def to_int16_array(
    array: np.ndarray, out: ndarrays_pb2.Int16Array | None = None
) -> ndarrays_pb2.Int16Array:
    """Populate a new or provided Int16Array message with given numpy array.

    Supports any int type taking <=4 bytes. If given e.g. an 8-bit or 16-bit
    integer, it will recast (sign extend) as a 16-bit integer and serialize.

    Note that this only accepts signed integers.

    Args:
        array: Array to add.
        out: Message to add to.
    """
    if out is None:
        out = ndarrays_pb2.Int16Array()
    endianness, dtype = _to_dtype(array.dtype, "i2")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_int16_array(msg: ndarrays_pb2.Int16Array) -> np.ndarray:
    """Convert a Int16Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with signed integer type.
    """
    return _from_int_array(msg, "i2")


def to_uint8_array(
    array: np.ndarray, out: ndarrays_pb2.UInt8Array | None = None
) -> ndarrays_pb2.UInt8Array:
    """Populate a new or provided UInt8Array message with given numpy array.

    Note that this only accepts unsigned integers.

    Args:
        array: Array to add.
        out: Message to add to.
    """
    if out is None:
        out = ndarrays_pb2.UInt8Array()
    _endianness, dtype = _to_dtype(array.dtype, "u1")
    out.shape[:] = array.shape
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_uint8_array(msg: ndarrays_pb2.UInt8Array) -> np.ndarray:
    """Convert a UInt8Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with unsigned integer type.
    """
    return _from_uint_array(msg, "u1")


def to_int8_array(
    array: np.ndarray, out: ndarrays_pb2.Int8Array | None = None
) -> ndarrays_pb2.Int8Array:
    """Populate a new or provided Int8Array message with given numpy array.

    Args:
        array: Array to add.
        out: Message to add to.
    """
    if out is None:
        out = ndarrays_pb2.Int8Array()
    _endianness, dtype = _to_dtype(array.dtype, "i1")
    out.shape[:] = array.shape
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_int8_array(msg: ndarrays_pb2.Int8Array) -> np.ndarray:
    """Convert a Int8Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array with signed integer type.
    """
    return _from_int_array(msg, "i1")


def _from_int_array(msg: TIntMsg, dtype_base: npt.DTypeLike) -> np.ndarray:
    """Convert a Int16Array/Int32Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.
        dtype_size: Number of bytes per int.

    Returns:
        Numpy array with signed integer type.
    """
    if not msg.shape:
        raise ValueError(f"Cannot convert unset/empty {type(msg)} message to a numpy array.")

    array_format = msg.WhichOneof("data")
    if array_format != "flat_bytes":
        raise ValueError(
            "Error deserializing int ndarray proto: only flat_bytes format "
            f"supported; given {array_format}"
        )  # pragma: no cover

    dtype = _from_dtype(
        getattr(msg, "endianness", cast(ndarrays_pb2.Endianness, ndarrays_pb2.LITTLE_ENDIAN)),
        dtype_base,
    )
    flat = np.frombuffer(msg.flat_bytes, dtype=dtype)
    return np.reshape(flat, msg.shape)


def _from_uint_array(msg: TUIntMsg, dtype_base: npt.DTypeLike) -> np.ndarray:
    """Convert a uIntArray proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.
        dtype_size: Number of bytes per int.

    Returns:
        Numpy array with unsigned integer type.
    """
    if not msg.shape:
        raise ValueError("Cannot convert unset/empty UIntArray message to a numpy array.")

    array_format = msg.WhichOneof("data")
    if array_format != "flat_bytes":
        raise ValueError(
            "Error deserializing uint ndarray proto: only flat_bytes format "
            f"supported; given {array_format}"
        )  # pragma: no cover

    dtype = _from_dtype(
        getattr(msg, "endianness", cast(ndarrays_pb2.Endianness, ndarrays_pb2.LITTLE_ENDIAN)),
        dtype_base,
    )
    flat = np.frombuffer(msg.flat_bytes, dtype=dtype)
    return np.reshape(flat, msg.shape)


def to_complex128_array(
    array: np.ndarray, out: ndarrays_pb2.Complex128Array | None = None
) -> ndarrays_pb2.Complex128Array:
    """Populate a new or provided Complex128Array msg with given numpy array.

    Supports any complex type taking <= 16 bytes. If given e.g. a c16/64-bit
    complex of single floats, this will recast as a 128-bit complex consisting
    of two 64-bit floats and serialize.

    Args:
        array: Array to add.
        out: Message to add to. If None, create new message.
    """
    if out is None:
        out = ndarrays_pb2.Complex128Array()
    endianness, dtype = _to_dtype(array.dtype, "c16")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_complex128_array(msg: ndarrays_pb2.Complex128Array) -> np.ndarray:
    """Convert a Complex128Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy complex array.
    """
    return _from_complex_array(msg, "c16")


def to_complex64_array(
    array: np.ndarray, out: ndarrays_pb2.Complex64Array | None = None
) -> ndarrays_pb2.Complex64Array:
    """Populate a new or provided Complex64Array msg with given numpy array.

    Supports any complex type taking <= 8 bytes. If given e.g. a c8/32-bit
    complex of single floats, this will recast as a 64-bit complex consisting
    of two 32-bit floats and serialize.

    Args:
        array: Array to add.
        out: Message to add to. If None, create new message.
    """
    if out is None:
        out = ndarrays_pb2.Complex64Array()
    endianness, dtype = _to_dtype(array.dtype, "c8")
    out.shape[:] = array.shape
    out.endianness = endianness
    out.flat_bytes = array.astype(dtype, copy=False).tobytes()
    return out


def from_complex64_array(msg: ndarrays_pb2.Complex64Array) -> np.ndarray:
    """Convert a Complex64Array proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy complex array.
    """
    return _from_complex_array(msg, "c8")


def _from_complex_array(msg: TComplexMsg, dtype_base: npt.DTypeLike) -> np.ndarray:
    if not msg.shape:
        raise ValueError(f"Cannot convert unset/empty {type(msg)} message to a numpy array.")

    array_format = msg.WhichOneof("data")
    if array_format != "flat_bytes":
        raise ValueError(
            "Error deserializing complex64 ndarray proto: only flat_bytes "
            f"format supported; given {array_format}"
        )  # pragma: no cover

    dtype = _from_dtype(cast(ndarrays_pb2.Endianness, msg.endianness), dtype_base)
    flat = np.frombuffer(msg.flat_bytes, dtype=dtype)
    return np.reshape(flat, msg.shape)


def to_bitarray(
    array: np.ndarray, out: ndarrays_pb2.BitArray | None = None
) -> ndarrays_pb2.BitArray:
    """Populate a new or provided BitArray message with the given numpy array.

    Supports bool or uint8 arrays. If given a uint8 array where values are not
    either 0 or 1 a ValueError will be raised.

    Args:
        array: Array to add.
        out: Message to add to. If None, create new message.
    """

    if out is None:
        out = ndarrays_pb2.BitArray()

    if not isinstance(array.dtype, bool):  # Includes np.bool
        # Validate that values are either 0 or 1.
        ok = np.logical_or(array == 0, array == 1)
        if not np.all(ok):
            # Determine which indices, and summarize those values.
            not_ok = array[np.where(~ok)]
            raise ValueError(
                f"Cannot convert array to bitarray; values must be either 0 or 1. Got {not_ok}"
            )

    # Add the numpy shape to the proto.
    out.shape[:] = array.shape
    out.flat_bytes = np.packbits(array).tobytes()
    return out


def from_bitarray(msg: ndarrays_pb2.BitArray) -> np.ndarray:
    """Convert a BitArray proto to a numpy array of the same shape.

    Args:
        msg: Message to extract array from.

    Returns:
        Numpy array.
    """
    if not msg.shape:
        raise ValueError(f"Cannot convert unset/empty {type(msg)} message to a numpy array.")

    # Load into uint8 array.
    flat_bytes = np.frombuffer(msg.flat_bytes, dtype=np.uint8)
    # Now extract flattened array from bits.
    tot_bits = functools.reduce(lambda x, y: x * y, msg.shape)
    flat_bits = np.unpackbits(flat_bytes, count=tot_bits)
    return np.reshape(flat_bits, msg.shape).view(np.bool_)

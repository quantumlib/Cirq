# Copyright 2018 The Cirq Developers
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

"""Tests for mem_manager."""

from __future__ import absolute_import

import multiprocessing
import pytest

import numpy as np

from cirq.google.sim import mem_manager


def get_shared_mem(handle: int) -> np.ndarray:
    return mem_manager.SharedMemManager.get_array(handle)


def test_create_supported_dtypes():
    for dtype in [
        np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32,
        np.float64
    ]:
        arr = np.array([1, 1], dtype=dtype)
        handle = mem_manager.SharedMemManager.create_array(arr)
        assert handle is not None
        np.testing.assert_equal(arr,
                                mem_manager.SharedMemManager.get_array(handle))
        mem_manager.SharedMemManager.free_array(handle)


def test_create_unsupported_dtype():
    with pytest.raises(ValueError) as exp:
        complex_arr = np.array([1j])
        mem_manager.SharedMemManager.create_array(complex_arr)
    assert 'dtype' in str(exp.value)


def test_create_unsupported_type():
    with pytest.raises(ValueError):
        not_an_array = 'not a numpy array'
        mem_manager.SharedMemManager.create_array(not_an_array)


def test_create_more_than_initial_allocation():
    handles = []
    for _ in range(1025):
        one = np.array([1])
        handles.append(mem_manager.SharedMemManager.create_array(one))
    for handle in handles:
        mem_manager.SharedMemManager.free_array(handle)


def test_fills_gaps():
    handles = []
    for _ in range(1024):
        one = np.array([1])
        handles.append(mem_manager.SharedMemManager.create_array(one))
    handle = handles.pop()
    mem_manager.SharedMemManager.free_array(handle)

    new_one = np.array([1])
    new_handle = mem_manager.SharedMemManager.create_array(new_one)

    for handle in handles:
        mem_manager.SharedMemManager.free_array(handle)
    mem_manager.SharedMemManager.free_array(new_handle)


def test_with_multiprocessing_pool():
    one = np.array([1])
    handle = mem_manager.SharedMemManager.create_array(one)
    pool = multiprocessing.Pool(processes=2)
    result = pool.map(get_shared_mem, [handle] * 10)
    pool.close()
    pool.join()
    np.testing.assert_equal([1] * 10, result)
    mem_manager.SharedMemManager.free_array(handle)


def test_with_multiple_multiprocessing_pools():
    one = np.array([1])
    two = np.array([2])
    one_handle = mem_manager.SharedMemManager.create_array(one)
    two_handle = mem_manager.SharedMemManager.create_array(two)
    one_pool = multiprocessing.Pool(processes=2)
    two_pool = multiprocessing.Pool(processes=2)
    one_result = one_pool.map(get_shared_mem, [one_handle] * 10)
    two_result = two_pool.map(get_shared_mem, [two_handle] * 10)
    one_pool.close()
    one_pool.join()
    two_pool.close()
    two_pool.join()
    np.testing.assert_equal([1] * 10, one_result)
    np.testing.assert_equal([2] * 10, two_result)
    mem_manager.SharedMemManager.free_array(one_handle)
    mem_manager.SharedMemManager.free_array(two_handle)

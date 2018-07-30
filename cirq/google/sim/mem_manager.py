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

"""Global level manager of shared numpy arrays."""
from typing import TYPE_CHECKING

import warnings
from multiprocessing import Lock, RawArray  # type: ignore

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Any, Optional, Tuple, List


class SharedMemManager(object):
    """Manager of global shared numpy arrays.

    Multiprocessing requires that shared memory needs to be inherited, and to
    use this with pools (not processes), this requires that it is global. This
    class is responsible for managing this global memory.
    """

    _INITIAL_SIZE = 1024

    _instance = None  # type: SharedMemManager

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SharedMemManager, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self._lock = Lock()
        self._current = 0
        self._count = 0
        self._arrays = SharedMemManager._INITIAL_SIZE * [
            None]  # type: List[Optional[Tuple[Any, Tuple[int, ...]]]]

    def _create_array(self, arr: np.ndarray) -> int:
        """Returns the handle of a RawArray created from the given numpy array.

        Args:
          arr: A numpy ndarray.

        Returns:
          The handle (int) of the array.

        Raises:
          ValueError: if arr is not a ndarray or of an unsupported dtype. If
            the array is of an unsupported type, using a view of the array to
            another dtype and then converting on get is often a work around.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError('Array is not a numpy ndarray.')
        try:
            c_arr = np.ctypeslib.as_ctypes(arr)
        except KeyError:
            raise ValueError(
                'Array has unsupported dtype {}.'.format(arr.dtype))

        # pylint: disable=protected-access
        raw_arr = RawArray(c_arr._type_, c_arr)

        with self._lock:
            if self._count >= len(self._arrays):
                self._arrays += len(self._arrays) * [None]

            self._get_next_free()

            # Note storing the shape is a workaround for an issue encountered
            # when upgrading to numpy 1.15.
            # See https://github.com/numpy/numpy/issues/11636
            self._arrays[self._current] = (raw_arr, arr.shape)

            self._count += 1

        return self._current

    def _get_next_free(self):
        loop_count = 0
        while self._arrays[self._current] is not None:
            self._current = (self._current + 1) % len(self._arrays)
            loop_count += 1
            if loop_count == len(self._arrays):
                raise RuntimeError(
                    'Cannot find free space to allocate new array.')

    def _free_array(self, handle: int):
        """Frees the memory for the array with the given handle.

        Args:
          handle: The handle of the array whose memory should be freed. This
            handle must come from the _create_array method.
        """
        with self._lock:
            if self._arrays[handle] is not None:
                self._arrays[handle] = None
                self._count -= 1

    def _get_array(self, handle: int) -> np.ndarray:
        """Returns the array with the given handle.

        Args:
          handle: The handle of the array whose memory should be freed. This
            handle must come from the _create_array method.

        Returns:
          The numpy ndarray with the handle given from _create_array.
        """
        tup = self._arrays[handle]
        assert tup is not None
        c_arr, shape = tup
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = np.ctypeslib.as_array(c_arr)
        result.shape = shape
        return result

    @staticmethod
    def get_instance() -> 'SharedMemManager':
        """Get the SharedMemManager instance."""
        if not SharedMemManager._instance:
            SharedMemManager._instance = SharedMemManager()
        return SharedMemManager._instance

    @staticmethod
    def create_array(arr: np.ndarray) -> int:
        """Returns the handle of a RawArray created from the given numpy array.

        Args:
          arr: A numpy ndarray. Only arrays with a dtype supported by numpy
            ctypeslib as_ctypes can be used.

        Returns:
          The handle (int) of the array.

        Raises:
          ValueError: if arr is not a ndarray or of an unsupported dtype. If
            the array is of an unsupported type, using a view of the array to
            another dtype and then converting on get is often a work around.
        """
        # pylint: disable=protected-access
        return SharedMemManager._instance._create_array(arr)

    @staticmethod
    def free_array(handle: int):
        """Frees the memory for the array with the given handle.

        Args:
          handle: The handle of the array whose memory should be freed. This
            handle must come from the create_array method.
        """
        # pylint: disable=protected-access
        SharedMemManager._instance._free_array(handle)

    @staticmethod
    def get_array(handle: int) -> np.ndarray:
        """Frees the memory for the array with the given handle.

        Args:
          handle: The handle of the array whose memory should be freed. This
            handle must come from the create_array method.

        Returns:
          The numpy ndarray with the handle given from _create_array.
        """
        # pylint: disable=protected-access
        return SharedMemManager._instance._get_array(handle)


# Create instance on module load.
SharedMemManager.get_instance()

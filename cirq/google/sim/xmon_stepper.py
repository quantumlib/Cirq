# Copyright 2018 Google LLC
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

"""Wavefunction simulator specialized to Google's xmon gate set.

This class should not be used directly, see instead Simulator in the
xmon_simulator class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import multiprocessing

from typing import Any, Dict, List, Union, Tuple

import numpy as np

from cirq.google.sim import mem_manager


I_PI_OVER_2 = 0.5j * np.pi


class Stepper(object):
    """A wave function simulator for quantum circuits with the xmon gate set.

    Xmons have a natural gate set made up of

    * Single qubit phase gates, exp(i t Z)
    * Single qubit gates about a operation in the Pauli X/Y plane,
      exp(i t (cos(theta) X + sin(theta)Y)
    * Two qubit phase gates exp(i t |11><11|)

    This stepper will do sharded simulation of the wave function using
    python's multiprocessing module.

    This stepper can be used like a context manager:
      with Stepper(num_qubits=3) as s:
        s.simulate_phases((1, 0.25))
        s.simulate_w(2, 0.25, 0.25)
        ...
    In this case the stepper will shut down the multiprocessing pool upon
    exiting the with context.

    If the  stepper is not used as a context manager, then it is required that
    __exit__ be called in order to ensure that the multiprocessing pool is
    properly closed (__enter__ does not need to be called).
    """

    def __init__(self,
        num_qubits: int,
        num_prefix_qubits: int = None,
        initial_state: Union[int, np.ndarray] = 0,
        min_qubits_before_shard: int = 13) -> None:
        """Construct a new Simulator.

        Args:
          num_qubits: The number of qubits to simulate.
          num_prefix_qubits: The wavefunction of the qubits is sharded into
            (2 ** num_prefix_qubits) parts. If this is None, then this will
            shard over the nearest power of two below the cpu count. If less
            than 10 qubits are being simulated then no sharding is done,
            depending on whether the shard_for_small_num_qubits is set or not.
          initial_state: If this is an int, then this is the state to initialize
            the stepper to, expressed as an integer of the computational basis.
            Integer to bitwise indices is little endian. Otherwise if this is
            a np.ndarray it is the full initial state and this must be the
            correct size, normalized (an L2 norm of 1), and have dtype of
            np.complex64.
          min_qubits_before_shard: Sharding will be done only for this number
            of qubits or more. The default is 13.
        """
        self._num_qubits = num_qubits
        if num_prefix_qubits is None:
            num_prefix_qubits = int(math.log(multiprocessing.cpu_count(), 2))
        if num_prefix_qubits > num_qubits:
            num_prefix_qubits = num_qubits
        if num_qubits < min_qubits_before_shard:
            num_prefix_qubits = 0
        self._num_prefix_qubits = num_prefix_qubits
        # Each shard is of a dimension equal to 2 ** num_shard_qubits.
        self._num_shard_qubits = self._num_qubits - self._num_prefix_qubits

        self._num_shards = 2 ** self._num_prefix_qubits
        self._shard_size = 2 ** self._num_shard_qubits

        # TODO(dabacon): This could be parallelized.
        self._init_shared_mem(initial_state)
        self._pool_open = False

    def _init_shared_mem(self, initial_state: int):
        self._shared_mem_dict = {}
        self.init_z_vects()
        self._init_scratch()
        self._init_state(initial_state)

    def init_z_vects(self):
        """Initializes bitwise vectors which is precomputed in shared memory.

        There are two types of vectors here, a zero one vectors and a pm
        (plus/minus) vectors. The pm vectors have rows that are Pauli Z
        operators acting on the all ones vector. The column-th row corresponds
        to the Pauli Z acting on the columnth-qubit.  Example for three shard
        qubits:
           [[1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1]]
        The zero one vectors are the pm vectors with 1 replacing -1 and 0
        replacing 1.

        There are number of shard qubit zero one vectors and each of these is of
        size equal to the shard size. For the zero one vectors, the ith one of
        these vectors has a  kth index value that is equal to 1 if the ith bit
        of k is set and zero otherwise. The vector directly encode the
        little-endian binary deigits of its index in the list:
        v[j][i] = (i >> j) & 1. For the pm vectors, the ith one of these vectors
        has a  kth index value that is equal to -1 if the ith bit of k is set
        and 1 otherwise.
        """
        shard_size = 2 ** self._num_shard_qubits

        a, b = np.indices((shard_size, self._num_shard_qubits))
        a >>= b
        a &= 1
        zero_one_vects = np.ascontiguousarray(a.transpose())
        zero_one_vects_handle = mem_manager.SharedMemManager.create_array(
            zero_one_vects)
        self._shared_mem_dict['zero_one_vects_handle'] = zero_one_vects_handle

        pm_vects = 1 - 2 * zero_one_vects
        pm_vects_handle = mem_manager.SharedMemManager.create_array(pm_vects)
        self._shared_mem_dict['pm_vects_handle'] = pm_vects_handle

    def _init_scratch(self):
        """Initializes a scratch pad equal in size to the wavefunction."""
        scratch = np.zeros((self._num_shards, self._shard_size),
                           dtype=np.complex64)
        scratch_handle = mem_manager.SharedMemManager.create_array(
            scratch.view(dtype=np.float32))
        self._shared_mem_dict['scratch_handle'] = scratch_handle

    def _init_state(self, initial_state: Union[int, np.ndarray]):
        """Initializes a the shard wavefunction and sets the initial state."""
        if isinstance(initial_state, int):
            state = np.zeros((self._num_shards, self._shard_size),
                             dtype=np.complex64)
            shard_num = initial_state // self._shard_size
            state[shard_num][initial_state % self._shard_size] = 1.0
        elif isinstance(initial_state, np.ndarray):
            self._check_state(initial_state)
            state = np.resize(initial_state,
                              (self._num_shards, self._shard_size))
        else:
            raise TypeError('Initial_state is not an int or np.ndarray.')

        state_handle = mem_manager.SharedMemManager.create_array(
            state.view(dtype=np.float32))
        self._shared_mem_dict['state_handle'] = state_handle

    def __del__(self):
        for handle in self._shared_mem_dict.values():
            mem_manager.SharedMemManager.free_array(handle)

    def __enter__(self):
        self._pool = (multiprocessing.Pool(processes=self._num_shards)
                      if self._num_qubits > 1 else ThreadlessPool())
        self._pool_open = True
        return self

    def __exit__(self, *args):
        self._pool.close()
        self._pool.join()
        self._pool_open = False

    def _shard_num_args(
        self, constant_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Helper that returns a list of dicts including a num_shard entry.

        The dict for each entry also includes shared_mem_dict, the number of
        shards, the number of shard qubits, and the supplied constant dict.

        Args:
          constant_dict: Dictionary that will be updated to every element of the
            returned list of dictionaries.

        Returns:
          A list of dictionaries. Each dictionary is constant except for the
          'shard_num' key which ranges from 0 to number of shards - 1.  Included
          keys are 'num_shards' and 'num_shard_qubits' along with all the
          keys in constant_dict.
        """
        args = []
        for shard_num in range(self._num_shards):
            append_dict = dict(constant_dict) if constant_dict else {}
            append_dict['shard_num'] = shard_num
            append_dict['num_shards'] = self._num_shards
            append_dict['num_shard_qubits'] = self._num_shard_qubits
            append_dict.update(self._shared_mem_dict)
            args.append(append_dict)
        return args

    @property
    def current_state(self):
        """Returns the current wavefunction."""
        # If the pool has been closed, recreate to calculate state.
        if not self._pool_open:
            self.__enter__()
        return np.array(self._pool.map(_state_shard,
                                       self._shard_num_args())).flatten()

    def reset_state(self, reset_state):
        """Reset the state to the given initial state.

        Args:
            reset_state: If this is an int, then this is the state to reset
            the stepper to, expressed as an integer of the computational basis.
            Integer to bitwise indices is little endian. Otherwise if this is
            a np.ndarray this must be the correct size, be normalized (L2 norm
            of 1), and have dtype of np.complex64.

        Raises:
            ValueError if the state is incorrectly sized or not of the correct
            dtype.
        """
        # If the pool has been closed, recreate to calculate state.
        if not self._pool_open:
            self.__enter__()
        if isinstance(reset_state, int):
            self._pool.map(_reset_state,
                           self._shard_num_args({'reset_state': reset_state}))
        elif isinstance(reset_state, np.ndarray):
            self._check_state(reset_state)
            args = []
            for kwargs in self._shard_num_args():
                shard_num = kwargs['shard_num']
                shard_size = 1 << kwargs['num_shard_qubits']
                start = shard_num * shard_size
                end = start + shard_size
                kwargs['reset_state'] = reset_state[start:end]
                args.append(kwargs)
            self._pool.map(_reset_state, args)

    def simulate_phases(self, phase_map: Dict[Tuple[int], float]):
        """Simulate a set of phase gates on the xmon architecture.

        Args:
          phase_map: A map from a tuple of indices to a value, one for each
            phase gate being simulated. If the tuple key has one index, then
            this is a Z phase gate on the index-th qubit with a rotation angle
            of pi times the value of the map. If the tuple key has two
            indices, then this is a |11> phasing gate, acting on the qubits at
            the two indices, and a rotation angle of pi times the value of
            the map.
        """
        if not self._pool_open:
            self.__enter__()
        self._pool.map(_clear_scratch, self._shard_num_args())
        # Iterate over the map of phase data.
        for indices, half_turns in phase_map.items():
            args = self._shard_num_args(
                {'indices': indices, 'half_turns': half_turns})
            if len(indices) == 1:
                self._pool.map(_single_qubit_accumulate_into_scratch, args)
            elif len(indices) == 2:
                self._pool.map(_two_qubit_accumulate_into_scratch, args)
        # Exponentiate the phases and add them into the state.
        self._pool.map(_apply_scratch_as_phase, self._shard_num_args())

    def simulate_w(self, index: int, half_turns: float,
        axis_half_turns: float):
        """Simulate a single qubit rotation gate about a X + b Y.

        The gate simulated is cos(pi half_turns) I + i sin (2 pi half_turns) *
        (cos (pi axis_half_turns) X + sin(pi axis_half_turns) Y)

        Args:
          index: The qubit to act on.
          half_turns: The amount of the overall rotation, see the formula above.
          axis_half_turns: The angle between the pauli X and Y operators,
            see the formula above.
        """
        if not self._pool_open:
            self.__enter__()
        args = self._shard_num_args({
            'index': index,
            'half_turns': half_turns,
            'axis_half_turns': axis_half_turns
        })
        if index >= self._num_shard_qubits:
            # W gate spans shards.
            self._pool.map(_clear_scratch, args)
            self._pool.map(_w_between_shards, args)
            self._pool.map(_copy_scratch_to_state, args)
        else:
            # W gate is within a shard.
            self._pool.map(_w_within_shard, args)

    def simulate_measurement(self, index: int) -> bool:
        """Simulates a single qubit measurement in the computational basis.

        Args:
          index: Which qubit is measured.

        Returns:
          True iff the measurement result corresponds to the |1> state.
        """
        if not self._pool_open:
            self.__enter__()
        args = self._shard_num_args({'index': index})
        prob_one = np.sum(self._pool.map(_one_prob_per_shard, args))
        result = (np.random.random() <= prob_one)

        args = self._shard_num_args({
            'index': index,
            'result': result,
            'prob_one': prob_one
        })
        self._pool.map(_collapse_state, args)
        return result

    def _check_state(self, state: np.ndarray):
        """Validates that the given state is a valid wave function."""
        if state.size != 1 << self._num_qubits:
            raise ValueError(
                'state state has incorrect size. Expected %s but was %s.' %
                (state.size, 1 << self._num_qubits))
        if state.dtype != np.complex64:
            raise ValueError(
                'Reset state has invalid dtype. Expected %s but was %s' % (
                    np.complex64, state.dtype))
        norm = np.sum(np.abs(state) ** 2)
        if not np.isclose(norm, 1):
            raise ValueError(
                'Initial state is not normalized instead had norm %s' % norm
            )



def _state_shard(args: Dict[str, Any]) -> np.ndarray:
    state_handle = args['state_handle']
    return mem_manager.SharedMemManager.get_array(state_handle).view(
        dtype=np.complex64)[args['shard_num']]


def _scratch_shard(args: Dict[str, Any]) -> np.ndarray:
    scratch_handle = args['scratch_handle']
    return mem_manager.SharedMemManager.get_array(scratch_handle).view(
        dtype=np.complex64)[args['shard_num']]


def _pm_vects(args: Dict[str, Any]) -> np.ndarray:
    return mem_manager.SharedMemManager.get_array(args['pm_vects_handle'])


def _zero_one_vects(args: Dict[str, Any]) -> np.ndarray:
    return mem_manager.SharedMemManager.get_array(args['zero_one_vects_handle'])


def _kth_bit(x: int, k: int) -> int:
    """Returns 1 if the kth bit of x is set, 0 otherwise."""
    return (x >> k) & 1


def _reset_state(args: Dict[str, Any]):
    shard_num = args['shard_num']
    shard_size = 2 ** args['num_shard_qubits']
    reset_state = args['reset_state']

    if isinstance(reset_state, int):
        _state_shard(args).fill(0)
        if shard_num == reset_state // shard_size:
            _state_shard(args)[reset_state % shard_size] = 1.0
    else:
        np.copyto(_state_shard(args), reset_state)



def _clear_scratch(args: Dict[str, Any]):
    """Sets all of the scratch shard to zero."""
    _scratch_shard(args).fill(0)


def _single_qubit_accumulate_into_scratch(args: Dict[str, Any]):
    """Accumuates single qubit phase gates into the scratch shards."""
    index = args['indices'][0]
    shard_num = args['shard_num']
    half_turns = args['half_turns']
    num_shard_qubits = args['num_shard_qubits']
    scratch = _scratch_shard(args)

    # ExpZ = exp(i pi Z half_turns / 2).
    if index >= num_shard_qubits:
        # Acts on prefix qubits.
        sign = 1 - 2 * _kth_bit(shard_num, index - num_shard_qubits)
        scratch += half_turns * sign
    else:
        # Acts on shard qubits.
        scratch += half_turns * _pm_vects(args)[index]


def _one_projector(args: Dict[str, Any], index: int) -> np.ndarray:
    """Returns a projector onto the |1> subspace of the index-th qubit."""
    num_shard_qubits = args['num_shard_qubits']
    shard_num = args['shard_num']
    if index >= num_shard_qubits:
        return _kth_bit(shard_num, index - num_shard_qubits)
    return _zero_one_vects(args)[index]


def _two_qubit_accumulate_into_scratch(args: Dict[str, Any]):
    """Accumulates two qubit phase gates into the scratch shards."""
    index0, index1 = args['indices']
    half_turns = args['half_turns']
    scratch = _scratch_shard(args)

    projector = _one_projector(args, index0) * _one_projector(args, index1)
    # Exp11 = exp(i pi |11><11| half_turns), but we accumulate phases as
    # pi / 2.
    scratch += 2 * half_turns * projector


def _apply_scratch_as_phase(args: Dict[str, Any]):
    """Takes scratch shards and applies them as exponentiated phase to state."""
    state = _state_shard(args)
    state *= np.exp(I_PI_OVER_2 * _scratch_shard(args))


def _w_within_shard(args: Dict[str, Any]):
    """Applies a W gate when the gate acts only within a shard."""
    index = args['index']
    half_turns = args['half_turns']
    axis_half_turns = args['axis_half_turns']
    state = _state_shard(args)
    pm_vect = _pm_vects(args)[index]
    num_shard_qubits = args['num_shard_qubits']
    shard_size = 2 ** num_shard_qubits

    reshape_tuple = (2 ** (num_shard_qubits - 1 - index), 2, 2 ** index)
    perm_state = np.reshape(
        np.reshape(state, reshape_tuple)[:, ::-1, :], shard_size)
    cos = np.cos(0.5 * np.pi * half_turns)
    sin = np.sin(0.5 * np.pi * half_turns)

    cos_axis = np.cos(np.pi * axis_half_turns)
    sin_axis = np.sin(np.pi * axis_half_turns)

    new_state = cos * state + 1j * sin * perm_state * (
        cos_axis - 1j * sin_axis * pm_vect)
    np.copyto(state, new_state)


def _w_between_shards(args: Dict[str, Any]):
    """Applies a W gate when the gate acts between shards."""
    shard_num = args['shard_num']
    state = _state_shard(args)
    num_shard_qubits = args['num_shard_qubits']
    index = args['index']
    half_turns = args['half_turns']

    axis_half_turns = args['axis_half_turns']

    perm_index = shard_num ^ (1 << (index - num_shard_qubits))
    perm_state = mem_manager.SharedMemManager.get_array(
        args['state_handle']).view(np.complex64)[perm_index]

    cos = np.cos(0.5 * np.pi * half_turns)
    sin = np.sin(0.5 * np.pi * half_turns)

    cos_axis = np.cos(np.pi * axis_half_turns)
    sin_axis = np.sin(np.pi * axis_half_turns)

    scratch = _scratch_shard(args)
    z_op = (1 - 2 * _kth_bit(shard_num, index - num_shard_qubits))
    np.copyto(scratch, state * cos + 1j * sin * perm_state *
              (cos_axis - 1j * sin_axis * z_op))


def _copy_scratch_to_state(args: Dict[str, Any]):
    """Copes scratch shards to state shards."""
    np.copyto(_state_shard(args), _scratch_shard(args))


def _one_prob_per_shard(args: Dict[str, Any]) -> float:
    """Returns the probability of getting a one measurement on a state shard."""
    index = args['index']

    state = _state_shard(args) * _one_projector(args, index)
    norm = np.linalg.norm(state)
    return norm * norm


def _collapse_state(args: Dict[str, Any]):
    """Projects state shards onto the appropriate post measurement state.

    This function makes no assumptions about the interpretation of quantum
    theory.

    Args:
      args: The args from shard_num_args.
    """
    index = args['index']
    result = args['result']
    prob_one = args['prob_one']

    state = _state_shard(args)
    normalization = np.sqrt(prob_one if result else 1 - prob_one)
    state *= (_one_projector(args, index) * result
        + (1 - _one_projector(args, index)) * (1 - result))
    state /= normalization


class ThreadlessPool(object):
    """A Pool that does not use any processes or threads.

    Only supports map, close, and join, the later two being trivial.
    No enforcement of closing or joining is done, so map can be called
    repeatedly.
    """

    def map(self, func, iterable, chunksize=None):
        assert chunksize is None, 'Chunking not supported by SimplePool'
        return [func(x) for x in iterable]

    def close(self):
        pass

    def join(self):
        pass

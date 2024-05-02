# Copyright 2024 The Cirq Developers
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

from typing import Sequence
from cirq_google.api.v2 import program_pb2
from cirq_google.api.v2 import run_context_pb2


# The special index of an empty directory path [].
_EMPTY_DIR_PATH_IDX = -1


def to_device_parameters_diff(
    device_params: Sequence[tuple[run_context_pb2.DeviceParameter, program_pb2.ArgValue]]
) -> run_context_pb2.DeviceParametersDiff:
    """Constructs a DeviceParametersDiff from multiple DeviceParameters and values

    Args:
        device_params: a list of (DeviceParameter, value) pairs.

    Returns:
        a DeviceParametersDiff, which comprises the entire device_params.
    """
    diff = run_context_pb2.DeviceParametersDiff()

    # Maps a string to its token id. A string is a component of a device parameter's path
    strs_index: dict[str, int] = {}

    def str_token_id(s: str) -> int:
        idx = strs_index.get(s)
        if idx is not None:
            return idx
        idx = len(diff.strs)
        strs_index[s] = idx
        diff.strs.append(s)
        return idx

    # Maps a directory path to its index into diff.dirs
    dirs_index: dict[tuple[str, ...], int] = {tuple(): _EMPTY_DIR_PATH_IDX}

    def dir_path_id(path: tuple[str, ...]) -> int:
        idx = dirs_index.get(path)
        if idx is not None:
            return idx
        # This path has not been seen. It will be appended to diff.dirs, with idx as,
        idx = len(diff.dirs)
        # Recursive call to get the assigned index of the parent. Note the base case
        # of the empty path, which returns -1.
        parent_id = dir_path_id(path[:-1])
        diff.dirs.add(parent=parent_id, name=str_token_id(path[-1]))
        dirs_index[path] = idx
        return idx

    for device_param, value in device_params:
        dir_path = tuple(device_param.path[:-1])
        param_name = device_param.path[-1]
        dir_id = dir_path_id(dir_path)
        diff.keys.add(name=str_token_id(param_name), dir=dir_id, value=value)

    return diff

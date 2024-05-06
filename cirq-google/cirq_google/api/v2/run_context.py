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

import functools
from typing import Sequence
from cirq_google.api.v2 import program_pb2
from cirq_google.api.v2 import run_context_pb2


# The special index of an empty directory path [].
_EMPTY_RESOURCE_PATH_IDX = -1


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

    # Maps a token to its token id. A token is a component of a device
    # parameter's path.
    strs_index: dict[str, int] = {}

    @functools.lru_cache(maxsize=2048)
    def token_id(s: str) -> int:
        idx = len(diff.strs)
        diff.strs.append(s)
        return idx

    # Maps a resource group path to its index in diff.groups.
    resource_groups_index: dict[tuple[str, ...], int] = {tuple(): _EMPTY_RESOURCE_PATH_IDX}

    def resource_path_id(path: tuple[str, ...]) -> int:
        idx = resource_groups_index.get(path)
        if idx is not None:
            return idx
        # This path has not been seen. It will be appended to diff.groups, with idx as,
        idx = len(diff.groups)
        # Recursive call to get the assigned index of the parent. Note the base case
        # of the empty path, which returns _EMPTY_RESOURCE_PATH_IDX.
        parent_id = resource_path_id(path[:-1])
        diff.groups.add(parent=parent_id, name=token_id(path[-1]))
        resource_groups_index[path] = idx
        return idx

    for device_param, value in device_params:
        resource_path = tuple(device_param.path[:-1])
        param_name = device_param.path[-1]
        path_id = resource_path_id(resource_path)
        diff.params.add(name=token_id(param_name), resource_group=path_id, value=value)

    return diff

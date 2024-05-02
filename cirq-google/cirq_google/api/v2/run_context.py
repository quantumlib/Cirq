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

    dirs_seen: set[tuple[int, int]] = set()

    for device_param, value in device_params:
        parent = -1  # no parent for the 1st path component
        for path_component in device_param.path[:-1]:
            token_id = str_token_id(path_component)
            if (parent, token_id) not in dirs_seen:
                diff.dirs.add(parent=parent, name=token_id)
                dirs_seen.add((parent, token_id))
            parent = token_id
        diff.keys.add(name=str_token_id(device_param.path[-1]), dir=parent, value=value)

    return diff

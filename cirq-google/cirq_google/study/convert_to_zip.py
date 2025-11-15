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
import cirq


def convert_to_zip(params: list[dict[str, float]]) -> cirq.Zip:
    """Converts a list of dictionaries into a cirq.Zip of cirq.Points.

    This will convert lists of dictionaries into a more compact
    Sweep format.   For large sweeps, this can vastly improve performance.

    This will change [{'a': 1.0, 'b': 2.0}, {'a': 3.0, 'b': 4.0}]
    into cirq.Zip(cirq.Points('a', [1.0, 3.0]), cirq.Points('b', [2.0, 4.0])_)

    Raises:
        ValueError if the keys in any of the list items are not the same.

    """
    param_keys: dict[str, list[float]] = {}
    if len(params) < 1:
        raise ValueError("Input dictionary to convert is empty.")
    sweep_keys = set(params[0].keys())
    for sweep_point in params:
        if set(sweep_point.keys()) != sweep_keys:
            raise ValueError("Keys must be the same in each sweep point.")
        for key, value in sweep_point.items():
            if (param_key_points := param_keys.get(key)) is None:
                param_keys[key] = []
                param_key_points = param_keys[key]
            param_key_points.append(value)
    return cirq.Zip(*[cirq.Points(key, points) for key, points in param_keys.items()])

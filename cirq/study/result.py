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

"""Defines trial results."""


class TrialResultMeta(type):
    """Metaclass that asserts measurements and params attributes exist."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if not hasattr(obj, 'params'):
            raise NotImplementedError(
                'TrialResult subclasses must have a params attribute.')
        if not hasattr(obj, 'measurements'):
            raise NotImplementedError(
                'TrialResult subclasses must have a measurements attribute.')
        return obj


class TrialResult(metaclass=TrialResultMeta):
    """The results of a single execution (trial).

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. The value for each key is a 2-D array of booleans, with
            the first index running over the repetitions, and the second index
            running over the qubits for the corresponding measurements.
    """

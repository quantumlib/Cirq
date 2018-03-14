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

"""Resolves ParameterValues to assigned values."""

from typing import Dict, Union

from cirq.study import ParameterizedValue


class ParamResolver(object):
    """Resolves a ParameterizedValue to an actual value.

    A ParameterizedValue is a parameter key (str) plus an offset float. A
    ParamResolver is an object that can be used to assign values for these keys.
    ParamResolvers are hashable.

    Attributes:
        param_dict: A dictionary from the ParameterValue key (str) to its
            assigned value.
    """

    def __init__(self, param_dict: Dict[str, float]) -> None:
        self.param_dict = param_dict
        self._param_hash = hash(frozenset(param_dict.items()))

    def value_of(self, parameterized_value: Union[ParameterizedValue, float]):
        """Resolves a ParameterizedValue to its assigned value.

        Args:
            parameterized_value: The ParameterizedValue to resolve or a float
                to be returned directly.

        Returns:
            The value of the parameter as resolved by this resolver. If the
            parameterized_value is just a float, then it will return this float.
            If the parameterized_value is a is a key plus a float, then this
            will return the assigned value for the key plus the float (offset).
        """
        if isinstance(parameterized_value, ParameterizedValue):
            return (
                self.param_dict[ParameterizedValue.key_of(parameterized_value)]
                + ParameterizedValue.val_of(parameterized_value))
        return parameterized_value

    def __hash__(self):
        return self._param_hash

    def __repr__(self):
        return 'ParamResolver({})'.format(repr(self.param_dict))

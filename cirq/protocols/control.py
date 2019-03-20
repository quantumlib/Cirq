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

from typing import Any, TypeVar, Union, List

import collections

import cirq

# This is a special indicator value used by the control method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ([],)  # type: Any


TDefault = TypeVar('TDefault')

def control(controllee: Union['cirq.Gate', 'cirq.OP_TREE'],
            control_qubits: List['cirq.Qid'] = None,
            default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns a Controlled version of the given value, if defined.

    Controllees define how to be controlled by defining a method
    __control__(self, control_qubits). Note that the method may return
    NotImplemented to indicate a particular controlling can't be done.

    Args:
        controllee: The gate, operation or iterable of operations to control.
        control_qubits: A list of Qids that would control this controllee.
        default: Determines the fallback behavior when `controllee` doesn't
            have a controlling defined. If `default` is not set and the
            fallback occurs, a TypeError is raised instead.

    Returns:
        If `controllee` has a __control__ method that returns something besides
        NotImplemented, that result is returned. For an OP_TREE, transformation
        is applied at the leaf. Otherwise, if a default value was specified,
        the default value is returned.

    Raises:
        TypeError: `controllee` doesn't have a __control__ method (or that
            method returned NotImplemented) and no `default` was specified.
    """
    controller = getattr(controllee, '__control__', None)
    result = NotImplemented if controller is None else controller(
                                                           control_qubits)
    if result is not NotImplemented:
        return result

    if isinstance(controllee, collections.Iterable):
        return cirq.transform_op_tree(controllee, op_transformation=
                                      lambda op: control(op, control_qubits))

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if controller is None:
        raise TypeError("object of type '{}' "
                        "has no __control__ method.".format(type(controllee)))
    raise TypeError("object of type '{}' does have a __control__ method, "
                    "but it returned NotImplemented.".format(type(controllee)))

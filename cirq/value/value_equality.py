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


def _value_equality_eq(self, other):
    cls_self = self._value_equality_values_cls_()
    if not isinstance(other, cls_self):
        return NotImplemented
    cls_other = other._value_equality_values_cls_()
    if cls_self != cls_other:
        return False
    return self._value_equality_values_() == other._value_equality_values_()


def _value_equality_ne(self, other):
    return not self == other


def _value_equality_hash(self):
    return hash((self._value_equality_values_cls_(),
                 self._value_equality_values_()))


def value_equality(*args,
                   unhashable: bool = False,
                   distinct_child_types: bool = False):
    """Implements __eq__, __ne__, and __hash__ via _value_equality_values_.

    Note that a type is appended to the equality values. By default this is the
    class type of the decorated class (so child types will be considered equal),
    but it can be changed to the type of the receiving value (so child type are
    all distinct). This logic is encoded by adding an appropriate
    `_value_equality_values_cls_` method to the class.

    Args:
        unhashable: When set, the __hash__ method will be set to None instead of
            to a hash of the equality class and equality values. Useful for
            mutable types such as dictionaries.
        distinct_child_types: When set, classes that inherit from the decorated
            class will not be considered equal to it. Also, different child
            classes will not be considered equal to each other. Useful for when
            the decorated class is an abstract class or trait that is helping to
            define equality for many conceptually distinct concrete classes.
    """

    # If `unhashable` was specified, the cls argument has not been passed yet.
    if len(args) == 0:
        return lambda cls: value_equality(
            cls,
            unhashable=unhashable,
            distinct_child_types=distinct_child_types)
    assert len(args) == 1

    cls = args[0]
    getter = getattr(cls, '_value_equality_values_', None)
    if getter is None:
        raise ValueError('The @cirq.value_equality decorator requires a '
                         '_value_equality_values_ method to be defined.')

    if distinct_child_types:
        setattr(cls, '_value_equality_values_cls_', lambda self: type(self))
    else:
        setattr(cls, '_value_equality_values_cls_', lambda self: cls)
    setattr(cls, '__hash__', None if unhashable else _value_equality_hash)
    setattr(cls, '__eq__', _value_equality_eq)
    setattr(cls, '__ne__', _value_equality_ne)

    return cls

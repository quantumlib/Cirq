# Copyright 2019 The Cirq Developers
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

"""A more flexible abstract base class metaclass ABCMetaAlt."""

import abc


def abstractmethod_alternatives(*alternative_list):
    """A decorator indicating an abstract method with a default implementation
    that uses any single function from alternative_list to implement default
    behavior.

    The metaclass must be ABCMetaAlt.  A class cannot be instantiated unless all
    abstract methods have been overridden.
    """
    def decorator(func):
        func._abstract_alternatives_ = alternative_list
        return func
    return decorator

# TODO: implement abstractproperty_, abstractstaticmethod_, and
# abstractclassmethod_alternatives


# This method is added to each class with the ABCMetaAlt metaclass.
def _alternative_for(cls_or_inst, name):
    """Returns the name of the alternative method to use in an implementation of
    an abstract method decorated with @abstract*_alternatives().

    Raises:
        AttributeError: The alternative method name does not exist in the class
            or instance."""
    try:
        alt_name = cls_or_inst._implemented_by_[name]
    except KeyError:
        raise AttributeError(
            'Alternative method named \'{}\' does not exist in {!r}'.format(
                name, cls_or_inst))
    return alt_name


class ABCMetaAlt(abc.ABCMeta):
    """A metaclass extending abc.ABCMeta for defining abstract base classes
    (ABCs) with more flexibility in which methods must be overridden.

    Use this metaclass in the same way as abc.ABCMeta to create an ABC.

    In addition to the decorators in the abc module, the decorator
    @abstractmethod_alternatives() can be used.
    """
    def __new__(mcls, name, bases, namespace, **kwargs):
        namespace['_alternative_for'] = _alternative_for
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        implemented_by = {}
        def has_some_implementation(name):
            if name in implemented_by:
                return implemented_by[name] is not None
            value = getattr(cls, name, None)
            if value is None:
                raise TypeError(
                    'A method named \'{}\' was listed as a possible '
                    'implementation alternative but it does not exist in the '
                    'definition of {!r}.'.format(
                        name, cls))
            if getattr(value, '__isabstractmethod__', False):
                return False
            if not hasattr(value, '_abstract_alternatives_'):
                return True
        def find_next_implementations(all_names):
            next_implemented_by = {}
            for name in all_names:
                if has_some_implementation(name):
                    continue
                value = getattr(cls, name, None)
                for alt_name in getattr(value, '_abstract_alternatives_'):
                    if has_some_implementation(alt_name):
                        next_implemented_by[name] = alt_name
                        break
            implemented_by.update(next_implemented_by)
            return bool(next_implemented_by)
        # Find all abstract methods (methods that havn't been implemented or
        # don't have an implemented alternative).
        all_names = set(namespace.keys())
        for base in bases:
            all_names.update(getattr(base, '__abstractmethods__', set()))
            all_names.update(getattr(base, '_implemented_by_', {}))
        while find_next_implementations(all_names):
            pass
        abstracts = frozenset(name
                              for name in all_names
                              if not has_some_implementation(name))
        # If __abstractmethods__ is non-empty, this is an abstract class and
        # can't be instantiated.
        cls.__abstractmethods__ |= abstracts  # Add to the set made by ABCMeta
        cls._implemented_by_ = implemented_by
        return cls

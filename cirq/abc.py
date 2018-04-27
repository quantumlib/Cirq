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

"""Shim class to allow using abstract base classes with mypy.

Use this instead of the standard-library abc module. It has the same effect as
the standard library module but gets around some mypy limitations.

In particular, mypy currently does not allow abstract base classes to be passed
to functions that accept Type objects as args, even in cases where the code is
valid and will not cause any type issues at runtime. We want to use this for the
Extension mechanism which allows to attempt to cast an object to a given
(possibly abstract) type.

Mypy's check for abstract types is very naive, so we can trick it by using this
module instead of importing abc directly from the standard library.

See https://github.com/python/mypy/issues/4717
"""

import abc as _abc

ABCMeta = _abc.ABCMeta
abstractmethod = _abc.abstractmethod
abstractproperty = _abc.abstractproperty

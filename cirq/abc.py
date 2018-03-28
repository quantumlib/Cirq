"""Shim class to allow using abstract base classes with mypy.

Use this instead of the standard-library abc module. It has the same effect as
the standard library module but gets around some mypy limitations.

In particular, mypy currently does not allow abstract base classes to be passed
to functions that accept Type objects as args, even in cases where the code is
valid and will not cause any type issues at runtime. We want to use this for the
Extension mechanism which allows to attempt to cast an object to a given
(possibly abstract) type.

See https://github.com/python/mypy/issues/4717
"""

import abc as _abc

ABCMeta = _abc.ABCMeta
abstractmethod = _abc.abstractmethod
abstractproperty = _abc.abstractproperty

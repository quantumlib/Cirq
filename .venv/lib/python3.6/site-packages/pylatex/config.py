# -*- coding: utf-8 -*-
"""
This module implements the ability to use of different configurations.

The current active configuration is `pylatex.config.active`. This variable can
simply be changed to another configuration and that will be used.

It is also possible to use the `~.Version1.use` method to do this temporarily
in a specific context.

..  :copyright: (c) 2016 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from contextlib import contextmanager


class Version1:
    """The config used to get the behaviour of v1.x.y of the library.

    The default attributes are::

        indent = True
        booktabs = False
        microtype = False
        row_height = None
    """

    indent = True
    booktabs = False
    microtype = False
    row_height = None

    def __init__(self, **kwargs):
        """
        Args
        ----
        kwargs:
            Key value pairs of the default attributes that should be overridden
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    @contextmanager
    def use(self):
        """Use the config temporarily in specific context.

        A simple usage example::

            with Version1(indent=False).use():
                # Do stuff where indent should be False
                ...


        """
        global active
        prev = active
        active = self
        yield
        active = prev

    @contextmanager
    def change(self, **kwargs):
        """Override some attributes of the config in a specific context.

        A simple usage example::

            with pylatex.config.active.change(indent=False):
                # Do stuff where indent should be False
                ...

        Args
        ----
        kwargs:
            Key value pairs of the default attributes that should be overridden
        """

        old_attrs = {}

        for k, v in kwargs.items():
            old_attrs[k] = getattr(self, k, v)
            setattr(self, k, v)

        yield self  # allows with ... as ...

        for k, v in old_attrs.items():
            setattr(self, k, v)


#: The default configuration in this release. Currently the same as `Version1`
Default = Version1


class Version2(Version1):
    """The config used to get the behaviour of v2.x.y of the library.

    The default attributes are::

        indent = False
        booktabs = True
        microtype = True
        row_height = 1.3
    """

    indent = False
    booktabs = True
    microtype = True
    row_height = 1.3

#: The default configuration in the nxt major release. Currently the same as
#: `Version2`.
NextMajor = Version2

#: The current active configuration
active = Default()

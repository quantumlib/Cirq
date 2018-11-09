# -*- coding: utf-8 -*-
"""
This module implements LaTeX base classes that can be subclassed.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from collections import UserList
from pylatex.utils import dumps_list
from contextlib import contextmanager
from .latex_object import LatexObject
from .command import Command, Arguments


class Container(LatexObject, UserList):
    """A base class that groups multiple LaTeX classes.

    This class should be subclassed when a LaTeX class has content that is of
    variable length. It subclasses UserList, so it holds a list of elements
    that can simply be accessed by using normal list functionality, like
    indexing or appending.

    """

    content_separator = '%\n'

    def __init__(self, *, data=None):
        r"""
        Args
        ----
        data: list, `~.LatexObject` or something that can be converted to a \
                string
            The content with which the container is initialized
        """

        if data is None:
            data = []
        elif not isinstance(data, list):
            # If the data is not already a list make it a list, otherwise list
            # operations will not work
            data = [data]

        self.data = data
        self.real_data = data  # Always the data of this instance

        super().__init__()

    @property
    def _repr_attributes(self):
        return super()._repr_attributes + ['real_data']

    def dumps_content(self, **kwargs):
        r"""Represent the container as a string in LaTeX syntax.

        Args
        ----
        \*\*kwargs:
            Arguments that can be passed to `~.dumps_list`


        Returns
        -------
        string:
            A LaTeX string representing the container
        """

        return dumps_list(self, escape=self.escape,
                          token=self.content_separator, **kwargs)

    def _propagate_packages(self):
        """Make sure packages get propagated."""

        for item in self.data:
            if isinstance(item, LatexObject):
                if isinstance(item, Container):
                    item._propagate_packages()
                for p in item.packages:
                    self.packages.add(p)

    def dumps_packages(self):
        r"""Represent the packages needed as a string in LaTeX syntax.

        Returns
        -------
        string:
            A LaTeX string representing the packages of the container
        """

        self._propagate_packages()

        return super().dumps_packages()

    @contextmanager
    def create(self, child):
        """Add a LaTeX object to current container, context-manager style.

        Args
        ----
        child: `~.Container`
            An object to be added to the current container
        """

        prev_data = self.data
        self.data = child.data  # This way append works appends to the child

        yield child  # allows with ... as to be used as well

        self.data = prev_data
        self.append(child)


class Environment(Container):
    r"""A base class for LaTeX environments.

    This class implements the basics of a LaTeX environment. A LaTeX
    environment looks like this:

    .. code-block:: latex

        \begin{environment_name}
            Some content that is in the environment
        \end{environment_name}

    The text that is used in the place of environment_name is by default the
    name of the class in lowercase.
    However, this default can be overridden in 2 ways:
    1. setting the _latex_name class variable when declaring the class
    2. setting the _latex_name attribute when initialising object
    """

    #: Set to true if this full container should be equivalent to an empty
    #: string if it has no content.
    omit_if_empty = False

    def __init__(self, *, options=None, arguments=None, start_arguments=None,
                 **kwargs):
        r"""
        Args
        ----
        options: str or list or  `~.Options`
            Options to be added to the ``\begin`` command

        arguments: str or list or `~.Arguments`
            Arguments to be added to the ``\begin`` command

        start_arguments: str or list or `~.Arguments`
            Arguments to be added before the options
        """

        self.options = options
        self.arguments = arguments
        self.start_arguments = start_arguments

        super().__init__(**kwargs)

    def dumps(self):
        """Represent the environment as a string in LaTeX syntax.

        Returns
        -------
        str
            A LaTeX string representing the environment.
        """

        content = self.dumps_content()
        if not content.strip() and self.omit_if_empty:
            return ''

        string = ''

        # Something other than None needs to be used as extra arguments, that
        # way the options end up behind the latex_name argument.
        if self.arguments is None:
            extra_arguments = Arguments()
        else:
            extra_arguments = self.arguments

        begin = Command('begin', self.start_arguments, self.options,
                        extra_arguments=extra_arguments)
        begin.arguments._positional_args.insert(0, self.latex_name)
        string += begin.dumps() + self.content_separator

        string += content + self.content_separator

        string += Command('end', self.latex_name).dumps()

        return string


class ContainerCommand(Container):
    r"""A base class for a container command (A command which contains data).

    Container command example:

    .. code-block:: latex

        \CommandName[options]{arguments}{
            data
        }

    """

    omit_if_empty = False

    def __init__(self, arguments=None, options=None, *, data=None, **kwargs):
        r"""
        Args
        ----
        arguments: str or `list`
            The arguments for the container command
        options: str, list or `~.Options`
            The options for the preamble command
        data: str or `~.LatexObject`
            The data to place inside the preamble command
        """

        self.arguments = arguments
        self.options = options

        super().__init__(data=data, **kwargs)

    def dumps(self):
        r"""Convert the container to a string in latex syntax."""

        content = self.dumps_content()

        if not content.strip() and self.omit_if_empty:
            return ''

        string = ''

        start = Command(self.latex_name, arguments=self.arguments,
                        options=self.options)

        string += start.dumps() + '{ \n'

        if content != '':
            string += content + '\n}'
        else:
            string += '}'

        return string

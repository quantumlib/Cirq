# -*- coding: utf-8 -*-
"""
This module implements the classes that deal with LaTeX lists.

These lists are specifically enumerate, itemize and description.

..  :copyright: (c) 2015 by Sean McLemon.
    :license: MIT, see License for more details.
"""

from .base_classes import Environment, Command, Options
from .package import Package
from pylatex.utils import NoEscape


class List(Environment):
    """A base class that represents a list."""

    #: List environments cause compile errors when they do not contain items.
    #: This is why they are omitted fully if they are empty.
    omit_if_empty = True

    def add_item(self, s):
        """Add an item to the list.

        Args
        ----
        s: str or `~.LatexObject`
            The item itself.
        """
        self.append(Command('item'))
        self.append(s)


class Enumerate(List):
    """A class that represents an enumerate list."""

    def __init__(self, enumeration_symbol=None, *, options=None, **kwargs):
        r"""
        Args
        ----
        enumeration_symbol: str
            The enumeration symbol to use, see the `enumitem
            <https://www.ctan.org/pkg/enumitem>`_ documentation to see what
            can be used here. This argument is not escaped as it usually
            should usually contain commands, so do not use user input here.
        options: str or list or `.Options`
            Custom options to be added to the enumerate list. These options are
            merged with the options created by ``enumeration_symbol``.
        """

        self._enumeration_symbol = enumeration_symbol

        if enumeration_symbol is not None:
            self.packages.add(Package("enumitem"))

            if options is not None:
                options = Options(options)
            else:
                options = Options()
            options._positional_args.append(NoEscape('label=' +
                                                     enumeration_symbol))

        super().__init__(options=options, **kwargs)


class Itemize(List):
    """A class that represents an itemize list."""


class Description(List):
    """A class that represents a description list."""

    def add_item(self, label, s):
        """Add an item to the list.

        Args
        ----
        label: str
            Description of the item.
        s: str or `~.LatexObject`
            The item itself.
        """
        self.append(Command('item', options=label))
        self.append(s)

# -*- coding: utf-8 -*-
"""
This module implements the base LaTeX object.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from ordered_set import OrderedSet
from ..utils import dumps_list
from abc import abstractmethod, ABCMeta
from reprlib import recursive_repr
from inspect import getfullargspec


class _CreatePackages(ABCMeta):
    def __init__(cls, name, bases, d):  # noqa
        packages = OrderedSet()

        for b in bases:
            if hasattr(b, 'packages'):
                packages |= b.packages

        if 'packages' in d:
            packages |= d['packages']

        cls.packages = packages

        super().__init__(name, bases, d)


class LatexObject(metaclass=_CreatePackages):
    """The class that every other LaTeX class is a subclass of.

    This class implements the main methods that every LaTeX object needs. For
    conversion to LaTeX formatted strings it implements the dumps, dump and
    generate_tex methods. It also provides the methods that can be used to
    represent the packages required by the LatexObject.
    """

    _latex_name = None
    _star_latex_name = False    # latex_name + ('*' if True else '')

    #: Set this to an iterable to override the list of default repr
    #: attributes.
    _repr_attributes_override = None
    #: Set this to a dict to change some of the default repr attributes to
    #: other attributes. The key is the old one, the value the new one.
    _repr_attributes_mapping = None

    #: Set on a class to make instances default to a certain kind of escaping
    _default_escape = True

    #: Only set this directly by changing the cls.escape
    _escape = None

    @property
    def escape(self):
        """Determine whether or not to escape content of this class.

        This defaults to `True` for most classes.
        """
        if self._escape is not None:
            return self._escape
        if self._default_escape is not None:
            return self._default_escape
        return True

    @escape.setter
    def escape(self, value):
        """Escape flag setter - to be used at object level."""
        self._escape = value

    #: Start a new paragraph before this environment.
    begin_paragraph = False

    #: Start a new paragraph after this environment.
    end_paragraph = False

    #: Same as enabling `begin_paragraph` and `end_paragraph`, so
    #: effectively placing this element in its own paragraph.
    separate_paragraph = False

    def __init__(self):
        # TODO: only create a copy of packages when it will
        # Create a copy of the packages attribute, so changing it in an
        # instance will not change the class default.
        self.packages = self.packages.copy()

    @recursive_repr()
    def __repr__(self):
        """Create a printable representation of the object."""

        return self.__class__.__name__ + '(' + \
            ', '.join(map(repr, self._repr_values)) + ')'

    @property
    def _repr_values(self):
        """Return values that are to be shown in repr string."""
        def getattr_better(obj, field):
            try:
                return getattr(obj, field)
            except AttributeError as e:
                try:
                    return getattr(obj, '_' + field)
                except AttributeError:
                    raise e

        return (getattr_better(self, attr) for attr in self._repr_attributes)

    @property
    def _repr_attributes(self):
        """Return attributes that should be part of the repr string."""
        if self._repr_attributes_override is None:
            # Default to init arguments
            attrs = getfullargspec(self.__init__).args[1:]
            mapping = self._repr_attributes_mapping
            if mapping:
                attrs = [mapping[a] if a in mapping else a for a in attrs]
            return attrs

        return self._repr_attributes_override

    @property
    def latex_name(self):
        """Return the name of the class used in LaTeX.

        It can be `None` when the class doesn't have a name.
        """
        star = ('*' if self._star_latex_name else '')
        if self._latex_name is not None:
            return self._latex_name + star
        return self.__class__.__name__.lower() + star

    @latex_name.setter
    def latex_name(self, value):
        self._latex_name = value

    @abstractmethod
    def dumps(self):
        """Represent the class as a string in LaTeX syntax.

        This method should be implemented by any class that subclasses this
        class.
        """

    def dump(self, file_w):
        """Write the LaTeX representation of the class to a file.

        Args
        ----
        file_w: io.TextIOBase
            The file object in which to save the data

        """

        file_w.write(self.dumps())

    def generate_tex(self, filepath):
        """Generate a .tex file.

        Args
        ----
        filepath: str
            The name of the file (without .tex)
        """

        with open(filepath + '.tex', 'w', encoding='utf-8') as newf:
            self.dump(newf)

    def dumps_packages(self):
        """Represent the packages needed as a string in LaTeX syntax.

        Returns
        -------
        list
        """

        return dumps_list(self.packages)

    def dump_packages(self, file_w):
        """Write the LaTeX representation of the packages to a file.

        Args
        ----
        file_w: io.TextIOBase
            The file object in which to save the data

        """

        file_w.write(self.dumps_packages())

    def dumps_as_content(self):
        """Create a string representation of the object as content.

        This is currently only used to add new lines before and after the
        output of the dumps function. These can be added or removed by changing
        the `begin_paragraph`, `end_paragraph` and `separate_paragraph`
        attributes of the class.
        """

        string = self.dumps()

        if self.separate_paragraph or self.begin_paragraph:
            string = '\n\n' + string.lstrip('\n')

        if self.separate_paragraph or self.end_paragraph:
            string = string.rstrip('\n') + '\n\n'

        return string

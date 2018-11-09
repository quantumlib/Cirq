# -*- coding: utf-8 -*-
"""
This module implements the classes that deal with math.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .base_classes import Command, Container, Environment
from .package import Package


class Alignat(Environment):
    """Class that represents a aligned equation environment."""

    #: Alignat environment cause compile errors when they do not contain items.
    #: This is why it is omitted fully if they are empty.
    omit_if_empty = True
    packages = [Package('amsmath')]

    def __init__(self, aligns=2, numbering=True, escape=None):
        """
        Parameters
        ----------
        aligns : int
            number of alignments
        numbering : bool
            Whether to number equations
        escape : bool
            if True, will escape strings
        """
        self.aligns = aligns
        self.numbering = numbering
        self.escape = escape
        if not numbering:
            self._star_latex_name = True
        super().__init__(start_arguments=[str(int(aligns))])


class Math(Container):
    """A class representing a math environment."""

    packages = [Package('amsmath')]

    content_separator = ' '

    def __init__(self, *, inline=False, data=None, escape=None):
        r"""
        Args
        ----
        data: list
            Content of the math container.
        inline: bool
            If the math should be displayed inline or not.
        escape : bool
            if True, will escape strings
        """

        self.inline = inline
        self.escape = escape
        super().__init__(data=data)

    def dumps(self):
        """Return a LaTeX formatted string representing the object.

        Returns
        -------
        str

        """
        if self.inline:
            return '$' + self.dumps_content() + '$'
        return '\\[%\n' + self.dumps_content() + '%\n\\]'


class VectorName(Command):
    """A class representing a named vector."""

    _repr_attributes_mapping = {
        'name': 'arguments',
    }

    def __init__(self, name):
        """
        Args
        ----
        name: str
            Name of the vector
        """

        super().__init__('mathbf', arguments=name)


class Matrix(Environment):
    """A class representing a matrix."""

    packages = [Package('amsmath')]

    _repr_attributes_mapping = {
        'alignment': 'arguments',
    }

    def __init__(self, matrix, *, mtype='p', alignment=None):
        r"""
        Args
        ----
        matrix: `numpy.ndarray` instance
            The matrix to display
        mtype: str
            What kind of brackets are used around the matrix. The different
            options and their corresponding brackets are:
            p = ( ), b = [ ], B = { }, v = \| \|, V = \|\| \|\|
        alignment: str
            How to align the content of the cells in the matrix. This is ``c``
            by default.

        References
        ----------
        * https://en.wikibooks.org/wiki/LaTeX/Mathematics#Matrices_and_arrays
        """

        import numpy  # noqa, Sanity check if numpy is installed

        self.matrix = matrix

        self.latex_name = mtype + 'matrix'
        self._mtype = mtype
        if alignment is not None:
            self.latex_name += '*'

        super().__init__(arguments=alignment)

    def dumps_content(self):
        """Return a string representing the matrix in LaTeX syntax.

        Returns
        -------
        str
        """

        import numpy as np

        string = ''
        shape = self.matrix.shape

        for (y, x), value in np.ndenumerate(self.matrix):
            if x:
                string += '&'
            string += str(value)

            if x == shape[1] - 1 and y != shape[0] - 1:
                string += r'\\' + '%\n'

        super().dumps_content()

        return string

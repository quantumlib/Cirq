# -*- coding: utf-8 -*-
"""
This module implements Error classes.

..  :copyright: (c) 2015 by Rene Beckmann.
    :license: MIT, see License for more details.
"""


class PyLaTeXError(Exception):
    """A Base class for all PyLaTeX Exceptions."""


class CompilerError(PyLaTeXError):
    """A Base class for all PyLaTeX compiler related Exceptions."""


class TableError(PyLaTeXError):
    """A Base class for all errors concerning tables."""


class TableRowSizeError(TableError):
    """Error for wrong table row size."""

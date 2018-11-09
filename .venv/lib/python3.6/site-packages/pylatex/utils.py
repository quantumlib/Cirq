# -*- coding: utf-8 -*-
"""
This module implements some simple utility functions.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

import os.path
import shutil
import tempfile
import pylatex.base_classes

_latex_special_chars = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\^{}',
    '\\': r'\textbackslash{}',
    '\n': '\\newline%\n',
    '-': r'{-}',
    '\xA0': '~',  # Non-breaking space
    '[': r'{[}',
    ']': r'{]}',
}

_tmp_path = os.path.abspath(
    os.path.join(
        tempfile.gettempdir(),
        "pylatex"
    )
)


def _is_iterable(element):
    return hasattr(element, '__iter__') and not isinstance(element, str)


class NoEscape(str):
    """
    A simple string class that is not escaped.

    When a `.NoEscape` string is added to another `.NoEscape` string it will
    produce a `.NoEscape` string. If it is added to normal string it will
    produce a normal string.

    Args
    ----
    string: str
        The content of the `NoEscape` string.
    """

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self)

    def __add__(self, right):
        s = super().__add__(right)
        if isinstance(right, NoEscape):
            return NoEscape(s)
        return s


def escape_latex(s):
    r"""Escape characters that are special in latex.

    Args
    ----
    s : `str`, `NoEscape` or anything that can be converted to string
        The string to be escaped. If this is not a string, it will be converted
        to a string using `str`. If it is a `NoEscape` string, it will pass
        through unchanged.

    Returns
    -------
    NoEscape
        The string, with special characters in latex escaped.

    Examples
    --------
    >>> escape_latex("Total cost: $30,000")
    'Total cost: \$30,000'
    >>> escape_latex("Issue #5 occurs in 30% of all cases")
    'Issue \#5 occurs in 30\% of all cases'
    >>> print(escape_latex("Total cost: $30,000"))

    References
    ----------
        * http://tex.stackexchange.com/a/34586/43228
        * http://stackoverflow.com/a/16264094/2570866
    """

    if isinstance(s, NoEscape):
        return s

    return NoEscape(''.join(_latex_special_chars.get(c, c) for c in str(s)))


def fix_filename(path):
    r"""Fix filenames for use in LaTeX.

    Latex has problems if there are one or more points in the filename, thus
    'abc.def.jpg' will be changed to '{abc.def}.jpg'

    Args
    ----
    filename : str
        The filen name to be changed.

    Returns
    -------
    str
        The new filename.

    Examples
    --------
    >>> fix_filename("foo.bar.pdf")
    '{foo.bar}.pdf'
    >>> fix_filename("/etc/local/foo.bar.pdf")
    '/etc/local/{foo.bar}.pdf'
    >>> fix_filename("/etc/local/foo.bar.baz/document.pdf")
    '/etc/local/foo.bar.baz/document.pdf'
    >>> fix_filename("/etc/local/foo.bar.baz/foo~1/document.pdf")
    '\detokenize{/etc/local/foo.bar.baz/foo~1/document.pdf}'
    """

    path_parts = path.split('/' if os.name == 'posix' else '\\')
    dir_parts = path_parts[:-1]

    filename = path_parts[-1]
    file_parts = filename.split('.')

    if len(file_parts) > 2:
        filename = '{' + '.'.join(file_parts[0:-1]) + '}.' + file_parts[-1]

    dir_parts.append(filename)
    fixed_path = '/'.join(dir_parts)

    if '~' in fixed_path:
        fixed_path = r'\detokenize{' + fixed_path + '}'

    return fixed_path


def dumps_list(l, *, escape=True, token='%\n', mapper=None, as_content=True):
    r"""Try to generate a LaTeX string of a list that can contain anything.

    Args
    ----
    l : list
        A list of objects to be converted into a single string.
    escape : bool
        Whether to escape special LaTeX characters in converted text.
    token : str
        The token (default is a newline) to separate objects in the list.
    mapper: callable or `list`
        A function, class or a list of functions/classes that should be called
        on all entries of the list after converting them to a string, for
        instance `~.bold` or `~.MediumText`.
    as_content: bool
        Indicates whether the items in the list should be dumped using
        `~.LatexObject.dumps_as_content`

    Returns
    -------
    NoEscape
        A single LaTeX string.

    Examples
    --------
    >>> dumps_list([r"\textbf{Test}", r"\nth{4}"])
    '\\textbf{Test}%\n\\nth{4}'
    >>> print(dumps_list([r"\textbf{Test}", r"\nth{4}"]))
    \textbf{Test}
    \nth{4}
    >>> print(pylatex.utils.dumps_list(["There are", 4, "lights!"]))
    There are
    4
    lights!
    >>> print(dumps_list(["$100%", "True"], escape=True))
    \$100\%
    True
    """
    strings = (_latex_item_to_string(i, escape=escape, as_content=as_content)
               for i in l)

    if mapper is not None:
        if not isinstance(mapper, list):
            mapper = [mapper]

        for m in mapper:
            strings = [m(s) for s in strings]
        strings = [_latex_item_to_string(s) for s in strings]

    return NoEscape(token.join(strings))


def _latex_item_to_string(item, *, escape=False, as_content=False):
    """Use the render method when possible, otherwise uses str.

    Args
    ----
    item: object
        An object that needs to be converted to a string
    escape: bool
        Flag that indicates if escaping is needed
    as_content: bool
        Indicates whether the item should be dumped using
        `~.LatexObject.dumps_as_content`

    Returns
    -------
    NoEscape
        Latex
    """

    if isinstance(item, pylatex.base_classes.LatexObject):
        if as_content:
            return item.dumps_as_content()
        else:
            return item.dumps()
    elif not isinstance(item, str):
        item = str(item)

    if escape:
        item = escape_latex(item)

    return item


def bold(s, *, escape=True):
    r"""Make a string appear bold in LaTeX formatting.

    bold() wraps a given string in the LaTeX command \textbf{}.

    Args
    ----
    s : str
        The string to be formatted.
    escape: bool
        If true the bold text will be escaped

    Returns
    -------
    NoEscape
        The formatted string.

    Examples
    --------
    >>> bold("hello")
    '\\textbf{hello}'
    >>> print(bold("hello"))
    \textbf{hello}
    """

    if escape:
        s = escape_latex(s)

    return NoEscape(r'\textbf{' + s + '}')


def italic(s, *, escape=True):
    r"""Make a string appear italicized in LaTeX formatting.

    italic() wraps a given string in the LaTeX command \textit{}.

    Args
    ----
    s : str
        The string to be formatted.
    escape: bool
        If true the italic text will be escaped

    Returns
    -------
    NoEscape
        The formatted string.

    Examples
    --------
    >>> italic("hello")
    '\\textit{hello}'
    >>> print(italic("hello"))
    \textit{hello}
    """
    if escape:
        s = escape_latex(s)

    return NoEscape(r'\textit{' + s + '}')


def verbatim(s, *, delimiter='|'):
    r"""Make the string verbatim.

    Wraps the given string in a \verb LaTeX command.

    Args
    ----
    s : str
        The string to be formatted.
    delimiter : str
        How to designate the verbatim text (default is a pipe | )

    Returns
    -------
    NoEscape
        The formatted string.

    Examples
    --------
    >>> verbatim(r"\renewcommand{}")
    '\\verb|\\renewcommand{}|'
    >>> print(verbatim(r"\renewcommand{}"))
    \verb|\renewcommand{}|
    >>> print(verbatim('pi|pe', '!'))
    \verb!pi|pe!
    """

    return NoEscape(r'\verb' + delimiter + s + delimiter)


def make_temp_dir():
    """Create a temporary directory if it doesn't exist.

    Directories created by this functionn follow the format specified
    by ``_tmp_path`` and are a pylatex subdirectory within
    a standard ``tempfile`` tempdir.

    Returns
    -------
    str
        The absolute filepath to the created temporary directory.

    Examples
    --------
    >>> make_temp_dir()
    '/var/folders/g9/ct5f3_r52c37rbls5_9nc_qc0000gn/T/pylatex'
    """

    if not os.path.exists(_tmp_path):
        os.makedirs(_tmp_path)
    return _tmp_path


def rm_temp_dir():
    """Remove the temporary directory specified in ``_tmp_path``."""

    if os.path.exists(_tmp_path):
        shutil.rmtree(_tmp_path)

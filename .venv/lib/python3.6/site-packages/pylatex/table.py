# -*- coding: utf-8 -*-
"""
This module implements the class that deals with tables.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .base_classes import LatexObject, Container, Command, UnsafeCommand, \
    Float, Environment
from .package import Package
from .errors import TableRowSizeError, TableError
from .utils import dumps_list, NoEscape, _is_iterable
import pylatex.config as cf

from collections import Counter
import re


# The letters used to count the table width
COLUMN_LETTERS = {'l', 'c', 'r', 'p', 'm', 'b', 'X'}


def _get_table_width(table_spec):
    """Calculate the width of a table based on its spec.

    Args
    ----
    table_spec: str
        The LaTeX column specification for a table.


    Returns
    -------
    int
        The width of a table which uses the specification supplied.
    """

    # Remove things like {\bfseries}
    cleaner_spec = re.sub(r'{[^}]*}', '', table_spec)

    # Remove X[] in tabu environments so they dont interfere with column count
    cleaner_spec = re.sub(r'X\[(.*?(.))\]', r'\2', cleaner_spec)
    spec_counter = Counter(cleaner_spec)

    return sum(spec_counter[l] for l in COLUMN_LETTERS)


class Tabular(Environment):
    """A class that represents a tabular."""

    _repr_attributes_mapping = {
        'table_spec': 'arguments',
        'pos': 'options',
    }

    def __init__(self, table_spec, data=None, pos=None, *,
                 row_height=None, col_space=None, width=None, booktabs=None,
                 **kwargs):
        """
        Args
        ----
        table_spec: str
            A string that represents how many columns a table should have and
            if it should contain vertical lines and where.
        pos: list
        row_height: float
            Specifies the heights of the rows in relation to the default
            row height
        col_space: str
            Specifies the spacing between table columns
        booktabs: bool
            Enable or disable booktabs style tables. These tables generally
            look nicer than regular tables. If this is `None` it will use the
            value of the ``booktabs`` attribte from the `~.active`
            configuration. This attribute is `False` by default.
        width: int
            The amount of columns that the table has. If this is `None` it is
            calculated based on the ``table_spec``, but this is only works for
            simple specs. In cases where this calculation is wrong override the
            width using this argument.

        References
        ----------
        * https://en.wikibooks.org/wiki/LaTeX/Tables#The_tabular_environment
        """

        if width is None:
            self.width = _get_table_width(table_spec)
        else:
            self.width = width

        if booktabs is None:
            booktabs = cf.active.booktabs
        self.booktabs = booktabs

        if self.booktabs:
            self.packages.add(Package('booktabs'))
            table_spec = '@{}%s@{}' % table_spec

        self.row_height = row_height if row_height is not None else \
            cf.active.row_height
        self.col_space = col_space

        super().__init__(data=data, options=pos,
                         arguments=NoEscape(table_spec),
                         **kwargs)

        # Parameter that determines if the xcolor package has been added.
        self.color = False

    def dumps(self):
        r"""Turn the Latex Object into a string in Latex format."""

        string = ""

        if self.row_height is not None:
            row_height = Command('renewcommand', arguments=[
                NoEscape(r'\arraystretch'),
                self.row_height])
            string += row_height.dumps() + '%\n'

        if self.col_space is not None:
            col_space = Command('setlength', arguments=[
                NoEscape(r'\tabcolsep'),
                self.col_space])
            string += col_space.dumps() + '%\n'

        return string + super().dumps()

    def dumps_content(self, **kwargs):
        r"""Represent the content of the tabular in LaTeX syntax.

        This adds the top and bottomrule when using a booktabs style tabular.

        Args
        ----
        \*\*kwargs:
            Arguments that can be passed to `~.dumps_list`

        Returns
        -------
        string:
            A LaTeX string representing the
        """

        content = ''
        if self.booktabs:
            content += '\\toprule%\n'

        content += super().dumps_content(**kwargs)

        if self.booktabs:
            content += '\\bottomrule%\n'

        return NoEscape(content)

    def add_hline(self, start=None, end=None, *, color=None,
                  cmidruleoption=None):
        r"""Add a horizontal line to the table.

        Args
        ----
        start: int
            At what cell the line should begin
        end: int
            At what cell the line should end
        color: str
            The hline color.
        cmidruleoption: str
            The option to be used for the booktabs cmidrule, i.e. the ``x`` in
            ``\cmidrule(x){1-3}``.
        """
        if self.booktabs:
            hline = 'midrule'
            cline = 'cmidrule'
            if cmidruleoption is not None:
                cline += '(' + cmidruleoption + ')'
        else:
            hline = 'hline'
            cline = 'cline'

        if color is not None:
            if not self.color:
                self.packages.append(Package('xcolor', options='table'))
                self.color = True
            color_command = Command(command="arrayrulecolor", arguments=color)
            self.append(color_command)

        if start is None and end is None:
            self.append(Command(hline))
        else:
            if start is None:
                start = 1
            elif end is None:
                end = self.width

            self.append(Command(cline,
                                dumps_list([start, NoEscape('-'), end])))

    def add_empty_row(self):
        """Add an empty row to the table."""

        self.append(NoEscape((self.width - 1) * '&' + r'\\'))

    def add_row(self, *cells, color=None, escape=None, mapper=None,
                strict=True):
        """Add a row of cells to the table.

        Args
        ----
        cells: iterable, such as a `list` or `tuple`
            There's two ways to use this method. The first method is to pass
            the content of each cell as a separate argument. The second method
            is to pass a single argument that is an iterable that contains each
            contents.
        color: str
            The name of the color used to highlight the row
        mapper: callable or `list`
            A function or a list of functions that should be called on all
            entries of the list after converting them to a string,
            for instance bold
        strict: bool
            Check for correct count of cells in row or not.
        """

        if len(cells) == 1 and _is_iterable(cells):
            cells = cells[0]

        if escape is None:
            escape = self.escape

        # Propagate packages used in cells
        for c in cells:
            if isinstance(c, LatexObject):
                for p in c.packages:
                    self.packages.add(p)

        # Count cell contents
        cell_count = 0

        for c in cells:
            if isinstance(c, MultiColumn):
                cell_count += c.size
            else:
                cell_count += 1

        if strict and cell_count != self.width:
            msg = "Number of cells added to table ({}) " \
                "did not match table width ({})".format(cell_count, self.width)
            raise TableRowSizeError(msg)

        if color is not None:
            if not self.color:
                self.packages.append(Package("xcolor", options='table'))
                self.color = True
            color_command = Command(command="rowcolor", arguments=color)
            self.append(color_command)

        self.append(dumps_list(cells, escape=escape, token='&',
                    mapper=mapper) + NoEscape(r'\\'))


class Tabularx(Tabular):
    """A class that represents a tabularx environment."""

    packages = [Package('tabularx')]

    def __init__(self, *args, width_argument=NoEscape(r'\textwidth'),
                 **kwargs):
        """
        Args
        ----
        width_argument:
            The width of the table. By default the table is as wide as the
            text.
        """
        super().__init__(*args, start_arguments=width_argument, **kwargs)


class MultiColumn(Container):
    """A class that represents a multicolumn inside of a table."""

    # TODO: Make this subclass of ContainerCommand

    def __init__(self, size, *, align='c', color=None, data=None):
        """
        Args
        ----
        size: int
            The amount of columns that this cell should fill.
        align: str
            How to align the content of the cell.
        data: str, list or `~.LatexObject`
            The content of the cell.
        color: str
            The color for the MultiColumn
        """

        self.size = size
        self.align = align

        super().__init__(data=data)

        # Add a cell color to the MultiColumn
        if color is not None:
            self.packages.append(Package('xcolor', options='table'))
            color_command = Command("cellcolor", arguments=color)
            self.append(color_command)

    def dumps(self):
        """Represent the multicolumn as a string in LaTeX syntax.

        Returns
        -------
        str
        """

        args = [self.size, self.align, self.dumps_content()]
        string = Command(self.latex_name, args).dumps()

        return string


class MultiRow(Container):
    """A class that represents a multirow in a table."""

    # TODO: Make this subclass CommandBase and Container

    packages = [Package('multirow')]

    def __init__(self, size, *, width='*', color=None, data=None):
        """
        Args
        ----
        size: int
            The amount of rows that this cell should fill.
        width: str
            Width of the cell. The default is ``*``, which means the content's
            natural width.
        data: str, list or `~.LatexObject`
            The content of the cell.
        color: str
            The color for the MultiRow
        """

        self.size = size
        self.width = width

        super().__init__(data=data)

        if color is not None:
            self.packages.append(Package('xcolor', options='table'))
            color_command = Command("cellcolor", arguments=color)
            self.append(color_command)

    def dumps(self):
        """Represent the multirow as a string in LaTeX syntax.

        Returns
        -------
        str
        """

        args = [self.size, self.width, self.dumps_content()]
        string = Command(self.latex_name, args).dumps()

        return string


class Table(Float):
    """A class that represents a table float."""


class Tabu(Tabular):
    """A class that represents a tabu (more flexible table)."""

    packages = [Package('tabu')]

    def __init__(self, table_spec, data=None, pos=None, *,
                 row_height=None, col_space=None, width=None, booktabs=None,
                 spread=None, to=None, **kwargs):
        """
        Args
        ----
        table_spec: str
            A string that represents how many columns a table should have and
            if it should contain vertical lines and where.
        pos: list
        row_height: float
            Specifies the heights of the rows in relation to the default
            row height
        col_space: str
            Specifies the spacing between table columns
        booktabs: bool
            Enable or disable booktabs style tables. These tables generally
            look nicer than regular tables. If this is `None` it will use the
            value of the ``booktabs`` attribte from the `~.active`
            configuration. This attribute is `False` by default.
        spread: str
            Specifies the Tabu table should add a given amount of 'padding' to
            the width of the table. This should be a latex dimension;
            for example: "0 pt" or "1in"
        to: str
            Specifies the Tabu table should extend to a given width.
            This should be a latex dimension; for example '4in'
        width: int
            The amount of columns that the table has. If this is `None` it is
            calculated based on the ``table_spec``, but this is only works for
            simple specs. In cases where this calculation is wrong override the
            width using this argument.

        References
        ----------
        * https://en.wikibooks.org/wiki/LaTeX/Tables#The_tabular_environment
        """

        super().__init__(table_spec, data, pos,
                         row_height=row_height, col_space=col_space,
                         width=width, booktabs=booktabs, **kwargs)

        self._preamble = ""
        if spread:
            self._preamble = "spread " + spread
        elif to:
            self._preamble = "to " + to

    def dumps(self):
        """Turn the tabu object into a string in Latex format."""

        _s = super().dumps()

        # Tabu tables support a unusual syntax:
        # \begin{tabu} spread 0pt {<col format...>}
        #
        # Since this syntax isn't common, it doesn't make
        # sense to support it in the baseclass (e.g., Environment)
        # rather, here we fix the LaTeX string post-hoc
        if self._preamble:
            if _s.startswith(r"\begin{longtabu}"):
                _s = _s[:16] + self._preamble + _s[16:]
            elif _s.startswith(r"\begin{tabu}"):
                _s = _s[:12] + self._preamble + _s[12:]
            else:
                raise TableError("Can't apply preamble to Tabu table "
                                 "(unexpected initial command sequence)")

        return _s


class LongTable(Tabular):
    """A class that represents a longtable (multipage table)."""

    packages = [Package('longtable')]

    header = False
    foot = False
    lastFoot = False

    def end_table_header(self):
        r"""End the table header which will appear on every page."""

        if self.header:
            msg = "Table already has a header"
            raise TableError(msg)

        self.header = True

        self.append(Command(r'endhead'))

    def end_table_footer(self):
        r"""End the table foot which will appear on every page."""

        if self.foot:
            msg = "Table already has a foot"
            raise TableError(msg)

        self.foot = True

        self.append(Command('endfoot'))

    def end_table_last_footer(self):
        r"""End the table foot which will appear on the last page."""

        if self.lastFoot:
            msg = "Table already has a last foot"
            raise TableError(msg)

        self.lastFoot = True

        self.append(Command('endlastfoot'))


class LongTabu(LongTable, Tabu):
    """A class that represents a longtabu (more flexible multipage table)."""


class LongTabularx(Tabularx, LongTable):
    """A class that represents a long version of the tabularx environment.

    This uses the ``ltablex`` package. This package modifies the ``tabularx``
    environment so that it can be spread over multiple pages. This has the
    sideeffect that using this class in a document spreads all `Tabularx`
    elements in that document over multiple pages as well.
    """

    _latex_name = 'tabularx'

    packages = [Package('ltablex')]


class ColumnType(UnsafeCommand):
    r"""A class representing a new column type.

    It uses the ``\newcolumntype`` command, for a thorough explanation see
    `this StackExchange question <https://tex.stackexchange.com/
    questions/257128/how-does-the-newcolumntype-command-work>`_.
    """

    _repr_attributes_mapping = {
        'name': 'arguments',
        'parameters': 'options'
    }

    def __init__(self, name, base, modifications, *, parameters=None):
        """
        Args
        ----
        name: str
            The name of the new column type (a single letter)
        base: str
            The name of the column type that the new one is based on (a single
            letter)
        modifications: str
            The modifications to be made to the base column type
        parameters: int
            The number of # parameters inside the modifications string, if this
            is `None` this is calculated automatically.
        """

        # repr vars
        self._base = base
        self._modifications = modifications

        COLUMN_LETTERS.add(name)

        if parameters is None:
            # count the number of non escaped #<number> parameters
            parameters = len(re.findall(r'(?<!\\)#\d', modifications))
            parameters += len(re.findall(r'(?<!\\)#\d', base))

        if parameters == 0:
            parameters = None

        modified = r">{%s\arraybackslash}%s" % (modifications, base)

        super().__init__(command="newcolumntype", arguments=name,
                         options=parameters, extra_arguments=modified)

"""
A library for creating Latex files.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .basic import HugeText, NewPage, LineBreak, NewLine, HFill, LargeText, \
    MediumText, SmallText, FootnoteText, TextColor
from .document import Document
from .frames import MdFramed, FBox
from .math import Math, VectorName, Matrix, Alignat
from .package import Package
from .section import Section, Subsection, Subsubsection
from .table import Table, MultiColumn, MultiRow, Tabular, Tabu, LongTable, \
    LongTabu, Tabularx, LongTabularx, ColumnType
from .tikz import TikZ, Axis, Plot, TikZNode, TikZDraw, TikZCoordinate, \
    TikZPathList, TikZPath, TikZUserPath, TikZOptions, TikZNodeAnchor, \
    TikZScope
from .figure import Figure, SubFigure, StandAloneGraphic
from .lists import Enumerate, Itemize, Description
from .quantities import Quantity
from .base_classes import Command, UnsafeCommand
from .utils import NoEscape, escape_latex
from .errors import TableRowSizeError
from .headfoot import PageStyle, Head, Foot, simple_page_number
from .position import Center, FlushLeft, FlushRight, MiniPage, TextBlock, \
    HorizontalSpace, VerticalSpace
from .labelref import Marker, Label, Ref, Pageref, Eqref, Autoref, Hyperref

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

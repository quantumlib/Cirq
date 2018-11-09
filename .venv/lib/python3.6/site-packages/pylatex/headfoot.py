# -*- coding: utf-8 -*-
"""
This module implements the classes that deal with creating headers and footers.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .base_classes import ContainerCommand, Command
from .package import Package
from .utils import NoEscape


class PageStyle(ContainerCommand):
    r"""Allows the creation of new page styles."""

    _latex_name = "fancypagestyle"

    packages = [Package('fancyhdr')]

    def __init__(self, name, *, header_thickness=0, footer_thickness=0,
                 data=None):
        r"""
        Args
        ----
        name: str
            The name of the page style
        header_thickness: float
            Value to set for the line under the header
        footer_thickness: float
            Value to set for the line over the footer
        data: str or `~.LatexObject`
            The data to place inside the PageStyle
        """

        self.name = name

        super().__init__(data=data, arguments=self.name)

        self.change_thickness(element="header", thickness=header_thickness)
        self.change_thickness(element="footer", thickness=footer_thickness)

        # Clear the current header and footer
        self.append(Head())
        self.append(Foot())

    def change_thickness(self, element, thickness):
        r"""Change line thickness.

        Changes the thickness of the line under/over the header/footer
        to the specified thickness.

        Args
        ----
        element: str
            the name of the element to change thickness for: header, footer
        thickness: float
            the thickness to set the line to
        """

        if element == "header":
            self.data.append(Command("renewcommand",
                             arguments=[NoEscape(r"\headrulewidth"),
                                        str(thickness) + 'pt']))
        elif element == "footer":
            self.data.append(Command("renewcommand", arguments=[
                NoEscape(r"\footrulewidth"), str(thickness) + 'pt']))


def simple_page_number():
    """Get a string containing commands to display the page number.

    Returns
    -------
    str
        The latex string that displays the page number
    """

    return NoEscape(r'Page \thepage\ of \pageref{LastPage}')


class Head(ContainerCommand):
    r"""Allows the creation of headers."""

    _latex_name = "fancyhead"

    def __init__(self, position=None, *, data=None):
        r"""
        Args
        ----
        position: str
            the headers position: L, C, R
        data: str or `~.LatexObject`
            The data to place inside the Head element
        """

        self.position = position

        super().__init__(data=data, options=position)


class Foot(Head):
    r"""Allows the creation of footers."""

    _latex_name = "fancyfoot"

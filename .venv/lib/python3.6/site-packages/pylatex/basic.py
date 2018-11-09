# -*- coding: utf-8 -*-
"""
This module implements several classes that represent basic latex commands.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .base_classes import CommandBase, Environment, ContainerCommand
from .package import Package


class NewPage(CommandBase):
    """A command that adds a new page to the document."""


class LineBreak(NewPage):
    """A command that adds a line break to the document."""


class NewLine(NewPage):
    """A command that adds a new line to the document."""


class HFill(NewPage):
    """A command that fills the current line in the document."""


class HugeText(Environment):
    """An environment which makes the text size 'Huge'."""

    _latex_name = "Huge"

    def __init__(self, data=None):
        """
        Args
        ----
        data : str or `~.LatexObject`
            The string or LatexObject to be formatted.
        """

        super().__init__(data=data)


class LargeText(HugeText):
    """An environment which makes the text size 'Large'."""

    _latex_name = "Large"


class MediumText(HugeText):
    """An environment which makes the text size 'large'."""

    _latex_name = "large"


class SmallText(HugeText):
    """An environment which makes the text size 'small'."""

    _latex_name = "small"


class FootnoteText(HugeText):
    """An environment which makes the text size 'footnotesize'."""

    _latex_name = "footnotesize"


class TextColor(ContainerCommand):
    """An environment which changes the text color of the data."""

    _repr_attributes_mapping = {
        "color": "arguments"
    }

    packages = [Package("xcolor")]

    def __init__(self, color, data):
        """
        Args
        ----
        color: str
            The color to set for the data inside of the environment.
        data: str or `~.LatexObject`
            The string or LatexObject to be formatted.
        """

        super().__init__(arguments=color, data=data)

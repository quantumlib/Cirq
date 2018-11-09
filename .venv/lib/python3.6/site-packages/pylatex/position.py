# -*- coding: utf-8 -*-
"""
This module implements the classes that deal with positioning.

Positions various elements on the page.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .base_classes import Environment, SpecialOptions, Command, CommandBase
from .package import Package
from .utils import NoEscape


class HorizontalSpace(CommandBase):
    """Add/remove the amount of horizontal space between elements."""

    _latex_name = 'hspace'

    _repr_attributes_mapping = {
        "size": "arguments"
    }

    def __init__(self, size, *, star=True):
        """
        Args
        ----
        size: str
            The amount of space to add
        star: bool
            Use the star variant of the command. Enabling this makes sure the
            space is also added where page breaking takes place.
        """

        if star:
            self.latex_name += '*'

        super().__init__(arguments=size)


class VerticalSpace(HorizontalSpace):
    """Add the user specified amount of vertical space to the document."""

    _latex_name = 'vspace'


class Center(Environment):
    r"""Centered environment."""

    packages = [Package('ragged2e')]


class FlushLeft(Center):
    r"""Left-aligned environment."""


class FlushRight(Center):
    r"""Right-aligned environment."""


class MiniPage(Environment):
    r"""A class that allows the creation of minipages within document pages."""

    packages = [Package('ragged2e')]

    _repr_attributes_mapping = {
        "width": "arguments",
        "pos": "options",
        "height": "options",
        "content_pos": "options",
        "align": "options"
    }

    def __init__(self, *, width=NoEscape(r'\textwidth'), pos=None,
                 height=None, content_pos=None, align=None, fontsize=None,
                 data=None):
        r"""
        Args
        ----
        width: str
            width of the minipage
        pos: str
            The vertical alignment of the minipage relative to the baseline
            (center(c), top(t), bottom(b))
        height: str
            height of the minipage
        content_pos: str
            The position of the content inside the minipage (center(c),
            bottom(b), top(t), spread(s))
        align: str
            alignment of the minibox
        fontsize: str
            The font size of the minipage
        data: str or `~.LatexObject`
            The data to place inside the MiniPage element
        """

        options = []

        if pos is not None:
            options.append(pos)

        if height is not None:
            options.append(NoEscape(height))

        if ((content_pos is not None) and (pos is not None) and
           (height is not None)):
            options.append(content_pos)

        options = SpecialOptions(*options)

        arguments = [NoEscape(str(width))]

        extra_data = []

        if align is not None:
            if align == "l":
                extra_data.append(Command(command="flushleft"))
            elif align == "c":
                extra_data.append(Command(command="centering"))
            elif align == "r":
                extra_data.append(Command(command="flushright"))

        if fontsize is not None:
            extra_data.append(Command(command=fontsize))

        if data is not None:
            if isinstance(data, list):
                data = extra_data + data
            else:
                data = extra_data + [data]
        else:
            data = extra_data

        super().__init__(arguments=arguments, options=options, data=data)


class TextBlock(Environment):
    r"""A class that represents a textblock environment.

    Make sure to set lengths of TPHorizModule and TPVertModule
    """

    _repr_attributes_mapping = {
        "width": "arguments"
    }

    packages = [Package('textpos')]

    def __init__(self, width, horizontal_pos, vertical_pos, *,
                 indent=False, data=None):
        r"""
        Args
        ----
        width: float
            Width of the text block in the units specified by TPHorizModule
        horizontal_pos: float
            Horizontal position in units specified by the TPHorizModule
        indent: bool
            Determines whether the text block has an indent before it
        vertical_pos: float
            Vertical position in units specified by the TPVertModule
        data: str or `~.LatexObject`
            The data to place inside the TextBlock element
        """

        arguments = width
        self.horizontal_pos = horizontal_pos
        self.vertical_pos = vertical_pos

        super().__init__(arguments=arguments)

        self.append("(%s, %s)" % (str(self.horizontal_pos),
                    str(self.vertical_pos)))

        if not indent:
            self.append(NoEscape(r'\noindent'))

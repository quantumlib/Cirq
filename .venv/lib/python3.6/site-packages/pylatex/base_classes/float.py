# -*- coding: utf-8 -*-
"""
This module implements the classes that deal with floating environments.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from . import Environment, Command


class Float(Environment):
    """A class that represents a floating environment."""

    #: By default floats are positioned inside a separate paragraph.
    #: Setting this to option to `False` will change that.
    separate_paragraph = True

    _repr_attributes_mapping = {
        'position': 'options',
    }

    def __init__(self, *, position=None, **kwargs):
        """
        Args
        ----
        position: str
            Define the positioning of a floating environment, for instance
            ``'h'``. See the references for more information.

        References
        ----------
            * https://www.sharelatex.com/learn/Positioning_of_Figures
        """

        super().__init__(options=position, **kwargs)

    def add_caption(self, caption):
        """Add a caption to the float.

        Args
        ----
        caption: str
            The text of the caption.
        """

        self.append(Command('caption', caption))

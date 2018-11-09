# -*- coding: utf-8 -*-
"""
This module implements the section type classes.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""


from .base_classes import Container, Command
from .labelref import Marker, Label


class Section(Container):
    """A class that represents a section."""

    #: A section should normally start in its own paragraph
    end_paragraph = True

    #: Default prefix to use with Marker
    marker_prefix = "sec"

    #: Number the sections when the section element is compatible,
    #: by changing the `~.Section` class default all
    #: subclasses will also have the new default.
    numbering = True

    def __init__(self, title, numbering=None, *, label=True, **kwargs):
        """
        Args
        ----
        title: str
            The section title.
        numbering: bool
            Add a number before the section title.
        label: Label or bool or str
            Can set a label manually or use a boolean to set
            preference between automatic or no label
        """

        self.title = title

        if numbering is not None:
            self.numbering = numbering
        if isinstance(label, Label):
            self.label = label
        elif isinstance(label, str):
            if ':' in label:
                label = label.split(':', 1)
                self.label = Label(Marker(label[1], label[0]))
            else:
                self.label = Label(Marker(label, self.marker_prefix))
        elif label:
            self.label = Label(Marker(title, self.marker_prefix))
        else:
            self.label = None

        super().__init__(**kwargs)

    def dumps(self):
        """Represent the section as a string in LaTeX syntax.

        Returns
        -------
        str

        """

        if not self.numbering:
            num = '*'
        else:
            num = ''

        string = Command(self.latex_name + num, self.title).dumps()
        if self.label is not None:
            string += '%\n' + self.label.dumps()
        string += '%\n' + self.dumps_content()

        return string


class Part(Section):
    """A class that represents a part."""

    marker_prefix = "part"


class Chapter(Section):
    """A class that represents a chapter."""

    marker_prefix = "chap"


class Subsection(Section):
    """A class that represents a subsection."""

    marker_prefix = "subsec"


class Subsubsection(Section):
    """A class that represents a subsubsection."""

    marker_prefix = "ssubsec"


class Paragraph(Section):
    """A class that represents a paragraph."""

    marker_prefix = "para"


class Subparagraph(Section):
    """A class that represents a subparagraph."""

    marker_prefix = "subpara"

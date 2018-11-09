# -*- coding: utf-8 -*-
"""
This module implements the class that deals with graphics.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

import posixpath

from .utils import fix_filename, make_temp_dir, NoEscape, escape_latex
from .base_classes import Float, UnsafeCommand
from .package import Package
import uuid


class Figure(Float):
    """A class that represents a Figure environment."""

    def add_image(self, filename, *, width=NoEscape(r'0.8\textwidth'),
                  placement=NoEscape(r'\centering')):
        """Add an image to the figure.

        Args
        ----
        filename: str
            Filename of the image.
        width: str
            The width of the image
        placement: str
            Placement of the figure, `None` is also accepted.

        """

        if width is not None:
            if self.escape:
                width = escape_latex(width)

            width = 'width=' + str(width)

        if placement is not None:
            self.append(placement)

        self.append(StandAloneGraphic(image_options=width,
                                      filename=fix_filename(filename)))

    def _save_plot(self, *args, extension='pdf', **kwargs):
        """Save the plot.

        Returns
        -------
        str
            The basename with which the plot has been saved.
        """
        import matplotlib.pyplot as plt

        tmp_path = make_temp_dir()
        filename = '{}.{}'.format(str(uuid.uuid4()), extension.strip('.'))
        filepath = posixpath.join(tmp_path, filename)

        plt.savefig(filepath, *args, **kwargs)
        return filepath

    def add_plot(self, *args, extension='pdf', **kwargs):
        """Add the current Matplotlib plot to the figure.

        The plot that gets added is the one that would normally be shown when
        using ``plt.show()``.

        Args
        ----
        args:
            Arguments passed to plt.savefig for displaying the plot.
        extension : str
            extension of image file indicating figure file type
        kwargs:
            Keyword arguments passed to plt.savefig for displaying the plot. In
            case these contain ``width`` or ``placement``, they will be used
            for the same purpose as in the add_image command. Namely the width
            and placement of the generated plot in the LaTeX document.
        """

        add_image_kwargs = {}

        for key in ('width', 'placement'):
            if key in kwargs:
                add_image_kwargs[key] = kwargs.pop(key)

        filename = self._save_plot(*args, extension=extension, **kwargs)

        self.add_image(filename, **add_image_kwargs)


class SubFigure(Figure):
    """A class that represents a subfigure from the subcaption package."""

    packages = [Package('subcaption')]

    #: By default a subfigure is not on its own paragraph since that looks
    #: weird inside another figure.
    separate_paragraph = False

    _repr_attributes_mapping = {
        'width': 'arguments',
    }

    def __init__(self, width=NoEscape(r'0.45\linewidth'), **kwargs):
        """
        Args
        ----
        width: str
            Width of the subfigure itself. It needs a width because it is
            inside another figure.

        """

        super().__init__(arguments=width, **kwargs)

    def add_image(self, filename, *, width=NoEscape(r'\linewidth'),
                  placement=None):
        """Add an image to the subfigure.

        Args
        ----
        filename: str
            Filename of the image.
        width: str
            Width of the image in LaTeX terms.
        placement: str
            Placement of the figure, `None` is also accepted.
        """

        super().add_image(filename, width=width, placement=placement)


class StandAloneGraphic(UnsafeCommand):
    r"""A class representing a stand alone image."""

    _latex_name = "includegraphics"

    packages = [Package('graphicx')]

    _repr_attributes_mapping = {
        "filename": "arguments",
        "image_options": "options"
    }

    def __init__(self, filename,
                 image_options=NoEscape(r'width=0.8\textwidth'),
                 extra_arguments=None):
        r"""
        Args
        ----
        filename: str
            The path to the image file
        image_options: str or `list`
            Specifies the options for the image (ie. height, width)
        """

        arguments = [NoEscape(filename)]

        super().__init__(command=self._latex_name, arguments=arguments,
                         options=image_options,
                         extra_arguments=extra_arguments)

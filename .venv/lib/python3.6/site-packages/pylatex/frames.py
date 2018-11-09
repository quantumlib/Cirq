# -*- coding: utf-8 -*-
"""
This module implements the classes that deal with adding frames.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .base_classes import Environment, ContainerCommand
from .package import Package


class MdFramed(Environment):
    """A class that defines an mdframed environment."""

    packages = [Package('mdframed')]


class FBox(ContainerCommand):
    """A class that defines an fbox ContainerCommand."""

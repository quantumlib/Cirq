"""
Baseclasses that can be used to create classes representing LaTeX objects.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

from .latex_object import LatexObject
from .containers import Container, Environment, ContainerCommand
from .command import CommandBase, Command, UnsafeCommand, Options, \
    SpecialOptions, Arguments
from .float import Float

# Old names of the base classes for backwards compatibility
BaseLaTeXClass = LatexObject
BaseLaTeXContainer = Container
BaseLaTeXNamedContainer = Environment

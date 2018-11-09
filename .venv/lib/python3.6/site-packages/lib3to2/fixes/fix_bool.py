"""
Fixer for __bool__ -> __nonzero__ methods.
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name

class FixBool(fixer_base.BaseFix):
    PATTERN = """
    classdef< 'class' any+ ':'
              suite< any*
                     funcdef< 'def' name='__bool__'
                              parameters< '(' NAME ')' > any+ >
                     any* > >
    """

    def transform(self, node, results):
        name = results["name"]
        new = Name("__nonzero__", prefix=name.prefix)
        name.replace(new)

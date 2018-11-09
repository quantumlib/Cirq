"""
Fixer for print: from __future__ import print_function.
"""

from lib2to3 import fixer_base
from lib3to2.fixer_util import future_import

class FixPrintfunction(fixer_base.BaseFix):

    explicit = True # Not the preferred way to fix print

    PATTERN = """
              power< 'print' trailer < '(' any* ')' > any* >
              """
    def transform(self, node, results):
        future_import("print_function", node)

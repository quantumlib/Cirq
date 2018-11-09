"""
Fixer for memoryview(s) -> buffer(s).
Explicit because some memoryview methods are invalid on buffer objects.
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name


class FixMemoryview(fixer_base.BaseFix):

    explicit = True # User must specify that they want this.

    PATTERN = """
              power< name='memoryview' trailer< '(' [any] ')' >
              rest=any* >
              """

    def transform(self, node, results):
        name = results["name"]
        name.replace(Name("buffer", prefix=name.prefix))

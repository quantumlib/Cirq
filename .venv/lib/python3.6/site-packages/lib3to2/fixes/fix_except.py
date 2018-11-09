"""
Fixer for except E as T -> except E, T
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Comma

class FixExcept(fixer_base.BaseFix):

    PATTERN = """except_clause< 'except' any as='as' any >"""

    def transform(self, node, results):
        results["as"].replace(Comma())

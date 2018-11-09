"""
Fixer for getfullargspec -> getargspec
"""

from lib2to3 import fixer_base
from ..fixer_util import Name

warn_msg = "some of the values returned by getfullargspec are not valid in Python 2 and have no equivalent."

class FixFullargspec(fixer_base.BaseFix):
    
    PATTERN = "'getfullargspec'"

    def transform(self, node, results):
        self.warning(node, warn_msg)
        return Name("getargspec", prefix=node.prefix)

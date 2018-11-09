"""
Fixer for:
str -> unicode
chr -> unichr
"spam" -> u"spam"
"""

import re
from lib2to3.pgen2 import token
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name

_mapping = {"chr": "unichr", "str": "unicode"}
_literal_re = re.compile(r"[rR]?[\'\"]")

class FixStr(fixer_base.BaseFix):

    order = "pre"
    run_order = 4 # Run this before bytes objects are converted to str objects

    PATTERN = "STRING | 'str' | 'chr'"

    def transform(self, node, results):
        new = node.clone()
        if node.type == token.STRING:
            # Simply add u to the beginning of the literal.
            if _literal_re.match(new.value):
                new.value = "u" + new.value
                return new
        elif node.type == token.NAME:
            assert new.value in _mapping
            new.value = _mapping[new.value]
            return new

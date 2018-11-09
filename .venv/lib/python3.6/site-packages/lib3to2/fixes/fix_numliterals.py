"""
Fixer for:
0b111101101 -> __builtins__.long("111101101", 2)
0o755 -> 0755
"""

import re

from lib2to3.pgen2 import token
from lib2to3 import fixer_base
from lib2to3.pygram import python_symbols as syms
from lib2to3.pytree import Node
from lib2to3.fixer_util import Number, Call, Attr, String, Name, ArgList, Comma

baseMAPPING = {'b':2, 'o':8, 'x':16}

def base(literal):
    """Returns the base of a valid py3k literal."""
    literal = literal.strip()
    # All literals that do not start with 0, or are 1 or more zeros.
    if not literal.startswith("0") or re.match(r"0+$",literal):
        return 10
    elif literal[1] not in "box":
        return 0
    return baseMAPPING[literal[1]]


class FixNumliterals(fixer_base.BaseFix):

    # We need to modify all numeric literals except floats, complex.
    def unmatch(self, node):
        """Don't match complex numbers, floats, or base-10 ints"""
        val = node.value
        for bad in "jJ+-.":
            if bad in val: return bad
        base_ = base(val)
        return base_ == 10 or base_ == 16

    def match(self, node):
        """Match number literals that are not excluded by self.unmatch"""
        return (node.type == token.NUMBER) and not self.unmatch(node)

    def transform(self, node, results):
        """
        Call __builtins__.long() with the value and the base of the value.
        This works because 0b10 is int("10", 2), 0o10 is int("10", 8), etc.
        """
        val = node.value
        base_ = base(val)
        if base_ == 8:
            assert val.strip().startswith("0o") or \
            val.strip().startswith("0O"), "Invalid format for octal literal"
            node.changed()
            node.value = "".join(("0",val[2:]))
        elif base_ == 2:
            assert val.startswith("0") and val[1] in "bB", \
                                           "Invalid format for binary literal"
            # __builtins__.long
            func_name = Node(syms.power, Attr(Name("__builtins__"), \
                             Name("long")))
            # ("...", 2)
            func_args = [String("".join(("\"", val.strip()[2:], "\""))), \
                         Comma(), Number(2, prefix=" ")]
            new_node = Call(func_name, func_args, node.prefix)
            return new_node

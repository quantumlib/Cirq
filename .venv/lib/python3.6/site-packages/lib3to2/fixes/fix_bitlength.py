"""
Fixer for:
anything.bit_length() -> (len(bin(anything)) - 2)
"""

from lib2to3 import fixer_base
from ..fixer_util import LParen, RParen, Call, Number, Name, Minus, Node, syms

class FixBitlength(fixer_base.BaseFix):

    PATTERN = "power< name=any trailer< '.' 'bit_length' > trailer< '(' ')' > >"

    def transform(self, node, results):
        
        name = results["name"]
        inner = Call(Name("bin"), [Name(name.value)])
        outer = Call(Name("len"), [inner])
        middle = Minus(prefix=" ")
        two = Number("2", prefix=" ")
        node.replace(Node(syms.power, [LParen(), outer, middle, two, RParen()], prefix=node.prefix))

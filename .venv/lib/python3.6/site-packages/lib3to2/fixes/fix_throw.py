"""Fixer for 'g.throw(E(V).with_traceback(T))' -> 'g.throw(E, V, T)'"""

from lib2to3 import fixer_base
from lib2to3.pytree import Node, Leaf
from lib2to3.pgen2 import token
from lib2to3.fixer_util import Comma

class FixThrow(fixer_base.BaseFix):

    PATTERN = """
    power< any trailer< '.' 'throw' >
        trailer< '(' args=power< exc=any trailer< '(' val=any* ')' >
        trailer< '.' 'with_traceback' > trailer< '(' trc=any ')' > > ')' > >
    """

    def transform(self, node, results):
        syms = self.syms
        exc, val, trc = (results["exc"], results["val"], results["trc"])
        val = val[0] if val else Leaf(token.NAME, "None")
        val.prefix = trc.prefix = " "
        kids = [exc.clone(), Comma(), val.clone(), Comma(), trc.clone()]
        args = results["args"]
        args.children = kids

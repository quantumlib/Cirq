"""
Fixer for dictcomp and setcomp:
{foo comp_for} -> set((foo comp_for))
{foo:bar comp_for} -> dict(((foo, bar) comp_for))"""

from lib2to3 import fixer_base
from lib2to3.pytree import Node, Leaf
from lib2to3.pygram import python_symbols as syms
from lib2to3.pgen2 import token
from lib2to3.fixer_util import parenthesize, Name, Call, LParen, RParen

from ..fixer_util import commatize

def tup(args):
    return parenthesize(Node(syms.testlist_gexp, commatize(args)))

class FixDctsetcomp(fixer_base.BaseFix):

    PATTERN = """atom< '{' dictsetmaker< 
                  n1=any [col=':' n2=any]
                    comp_for=comp_for< 'for' any 'in' any [comp_if<'if' any>] >
                  > '}' >"""

    def transform(self, node, results):
        comp_for = results.get("comp_for").clone()
        is_dict = bool(results.get("col")) # is it a dict?
        n1 = results.get("n1").clone()
        if is_dict:
            n2 = results.get("n2").clone()
            n2.prefix = " "
            impl_assign = tup((n1, n2))
        else:
            impl_assign = n1
        our_gencomp = Node(syms.listmaker, [(impl_assign),(comp_for)])
        if is_dict:
            new_node = Node(syms.power, [Name("dict"),
                       parenthesize(Node(syms.atom, [our_gencomp]))])
        else:
            new_node = Node(syms.power, [Name("set"),
                       parenthesize(Node(syms.atom, [our_gencomp]))])
        new_node.prefix = node.prefix
        return new_node

"""
Fixer to remove class decorators
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Assign, String, Newline
from ..fixer_util import Leaf, Node, token, syms, indentation

class FixClassdecorator(fixer_base.BaseFix):

    PATTERN = """
              decorated < one_dec=decorator < any* > cls=classdef < 'class' name=any any* > > |
              decorated < decorators < decs=decorator+ > cls=classdef < 'class' name=any any* > >
              """
    def transform(self, node, results):

        singleton = results.get("one_dec")
        classdef = results["cls"]
        decs = [results["one_dec"]] if results.get("one_dec") is not None else results["decs"]
        dec_strings = [str(dec).strip()[1:] for dec in decs]
        assign = ""
        for dec in dec_strings:
            assign += dec
            assign += "("
        assign += results["name"].value
        for dec in dec_strings:
            assign += ")"
        assign = String(results["name"].value + " = " + assign)
        assign_statement = Node(syms.simple_stmt, [assign, Newline(), Newline()])
        prefix = None
        for dec in decs:
            if prefix is None:
                prefix = dec.prefix
            dec.remove()
        classdef.prefix = prefix
        i = indentation(node)
        pos = node.children.index(classdef) + 1
        if classdef.children[-1].children[-1].type == token.DEDENT:
            del classdef.children[-1].children[-1]
        node.insert_child(pos, Leaf(token.INDENT, i))
        node.insert_child(pos, assign_statement)
        node.insert_child(pos, Leaf(token.INDENT, i))



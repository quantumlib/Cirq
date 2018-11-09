"""
Fixer for bytes -> str.
"""

import re
from lib2to3 import fixer_base
from lib2to3.patcomp import compile_pattern
from ..fixer_util import Name, token, syms, parse_args, Call, Comma

_literal_re = re.compile(r"[bB][rR]?[\'\"]")

class FixBytes(fixer_base.BaseFix):

    order = "pre"
    
    PATTERN = "STRING | power< 'bytes' [trailer< '(' (args=arglist | any*) ')' >] > | 'bytes'"

    def transform(self, node, results):
        name = results.get("name")
        arglist = results.get("args")
        if node.type == token.NAME:
            return Name("str", prefix=node.prefix)
        elif node.type == token.STRING:
            if _literal_re.match(node.value):
                new = node.clone()
                new.value = new.value[1:]
                return new
        if arglist is not None:
            args = arglist.children
            parsed = parse_args(args, ("source", "encoding", "errors"))

            source, encoding, errors = (parsed[v] for v in ("source", "encoding", "errors"))
            encoding.prefix = ""
            str_call = Call(Name("str"), ([source.clone()]))
            if errors is None:
                node.replace(Call(Name(str(str_call) + ".encode"), (encoding.clone(),)))
            else:
                errors.prefix = " "
                node.replace(Call(Name(str(str_call) + ".encode"), (encoding.clone(), Comma(), errors.clone())))

"""
Fixer for:

def something(self):
    super()

->

def something(self):
    super(self.__class__, self)
"""

from lib2to3 import fixer_base
from ..fixer_util import Node, Leaf, token, syms, Name, Comma, Dot

dot_class = Node(syms.trailer, [Dot(), Name("__class__")])

def get_firstparam(super_node):
    parent = super_node.parent
    while parent.type != syms.funcdef and parent.parent:
        parent = parent.parent

    if parent.type != syms.funcdef:
        # super() called without arguments outside of a funcdef
        return None

    children = parent.children
    assert len(children) > 2
    params = children[2]
    assert params.type == syms.parameters
    if len(params.children) < 3:
        # Function has no parameters, therefore super() makes no sense here...
        return None
    args = params.children[1]
    if args.type == token.NAME:
        return args.value
    elif args.type == syms.typedargslist:
        assert len(args.children) > 0
        if args.children[0].type == token.NAME:
            return args.children[0].value
        else:
            # Probably a '*'
            return None


def get_class_name(super_node):
    parent = super_node.parent
    while parent.type != syms.classdef and parent.parent:
        parent = parent.parent

    if parent.type != syms.classdef:
        # super() called without arguments outside of a classdef
        return None

    children = parent.children
    assert len(children) > 2
    class_name = children[1]
    assert class_name.type == token.NAME
    return class_name.value


def insert_args(name, class_name, rparen):
    parent = rparen.parent
    
    if class_name:
        class_node = Node(syms.power, [Name(class_name)])
    else:
        class_node = Node(syms.power, [Name(name), dot_class.clone()])

    idx = parent.children.index(rparen)
    parent.insert_child(idx, Name(name, prefix=" "))
    parent.insert_child(idx, Comma())
    parent.insert_child(idx, class_node)


class FixSuper(fixer_base.BaseFix):

    PATTERN = "power< 'super' trailer< '(' rparen=')' > any* >"

    def transform(self, node, results):
        param = get_firstparam(node)
        if param is None:
            self.cannot_convert(node, "super() with no arguments must be called inside a function that has at least one parameter")
            return
        class_name = get_class_name(node)
        rparen = results["rparen"]
        insert_args(param, class_name, rparen)

"""
Fixer for print function to print statement

print(spam,ham,eggs,sep=sep,end=end,file=file)
->
print >>file, sep.join((str(spam),str(ham),str(eggs))),; file.write(end)
in the most complicated case.  Simpler cases:
print() -> print
print("spam") -> print "spam"
print(1,2,3) -> print 1,2,3
print(1,2,3,end=" ") -> print 1,2,3,
print(1,2,3,end="") -> print 1,2,3,; sys.stdout.write("")
print(1,2,3,file=file) -> print >>file, 1,2,3
print(1,2,3,sep=" ",end="\n") -> print 1,2,3
"""

from __future__ import with_statement # Aiming for 2.5-compatible code

from lib2to3 import fixer_base
from lib2to3.pytree import Node, Leaf
from lib2to3.pygram import python_symbols as syms, token
from lib2to3.fixer_util import (Name, FromImport, Newline, Call, Comma, Dot,
                                LParen, RParen, touch_import)
import warnings
import sys

def gen_printargs(lst):
    """
    Accepts a list of all nodes in the print call's trailer.
    Yields nodes that will be easier to deal with
    """
    for node in lst:
        if node.type == syms.arglist:
            # arglist<pos=any* kwargs=(argument<"file"|"sep"|"end" "=" any>*)>
            kids = node.children
            it = kids.__iter__()
            try:
                while True:
                    arg = next(it)
                    if arg.type == syms.argument:
                        # argument < "file"|"sep"|"end" "=" (any) >
                        yield arg
                        next(it)
                    else:
                        yield arg
                        next(it)
            except StopIteration:
                continue
        else:
            yield node

def isNone(arg):
    """
    Returns True if arg is a None node
    """
    return arg.type == token.NAME and arg.value == "None"

def _unicode(arg):
    """
    Calls unicode() on the arg in the node.
    """
    prefix = arg.prefix
    arg = arg.clone()
    arg.prefix = ""
    ret = Call(Name("unicode", prefix=prefix), [arg])
    return ret

def add_file_part(file, lst):
    if file is None or isNone(file):
        return
    lst.append(Leaf(token.RIGHTSHIFT, ">>", prefix=" "))
    lst.append(file.clone())
    lst.append(Comma())

def add_sep_part(sep, pos, lst):
    if sep is not None and not isNone(sep) and \
       not (sep.type == token.STRING and sep.value in ("' '", '" "')):
        temp = []
        for arg in pos:
            temp.append(_unicode(arg.clone()))
            if sys.version_info >= (2, 6):
                warnings.warn("Calling unicode() on what may be a bytes object")
            temp.append(Comma())
        del temp[-1]
        sep = sep.clone()
        sep.prefix = " "
        args = Node(syms.listmaker, temp)
        new_list = Node(syms.atom, [Leaf(token.LSQB, "["), args,
                                    Leaf(token.RSQB, "]")])
        join_arg = Node(syms.trailer, [LParen(), new_list, RParen()])
        sep_join = Node(syms.power, [sep, Node(syms.trailer,
                                                      [Dot(), Name("join")])])
        lst.append(sep_join)
        lst.append(join_arg)
    else:
        if pos:
            pos[0].prefix = " "
        for arg in pos:
            lst.append(arg.clone())
            lst.append(Comma())
        del lst[-1]

def add_end_part(end, file, parent, loc):
    if isNone(end):
        return
    if end.type == token.STRING and end.value in ("' '", '" "',
                                                  "u' '", 'u" "',
                                                  "b' '", 'b" "'):
        return
    if file is None:
        touch_import(None, "sys", parent)
        file = Node(syms.power, [Name("sys"),
                                 Node(syms.trailer, [Dot(), Name("stdout")])])
    end_part = Node(syms.power, [file,
                                Node(syms.trailer, [Dot(), Name("write")]),
                                Node(syms.trailer, [LParen(), end, RParen()])])
    end_part.prefix = " "
    parent.insert_child(loc, Leaf(token.SEMI, ";"))
    parent.insert_child(loc+1, end_part)

def replace_print(pos, opts, old_node=None):
    """
    Replace old_node with a new statement.
    Also hacks in the "end" functionality.
    """
    new_node = new_print(*pos, **opts)
    end = None if "end" not in opts else opts["end"].clone()
    file = None if "file" not in opts else opts["file"].clone()
    if old_node is None:
        parent = Node(syms.simple_stmt, [Leaf(token.NEWLINE, "\n")])
        i = 0
    else:
        parent = old_node.parent
        i = old_node.remove()
    parent.insert_child(i, new_node)
    if end is not None and not (end.type == token.STRING and \
                               end.value in ("'\\n'", '"\\n"')):
        add_end_part(end, file, parent, i+1)
    return new_node

def new_print(*pos, **opts):
    """
    Constructs a new print_stmt node
    args is all positional arguments passed to print()
    kwargs contains zero or more of the following mappings:
    
    'sep': some string
    'file': some file-like object that supports the write() method
    'end': some string
    """
    children = [Name("print")]
    sep = None if "sep" not in opts else opts["sep"]
    file = None if "file" not in opts else opts["file"]
    end = None if "end" not in opts else opts["end"]
    add_file_part(file, children)
    add_sep_part(sep, pos, children)
    if end is not None and not isNone(end):
        if not end.value in ('"\\n"', "'\\n'"):
            children.append(Comma())
    return Node(syms.print_stmt, children)

def map_printargs(args):
    """
    Accepts a list of all nodes in the print call's trailer.
    Returns {'pos':[all,pos,args], 'sep':sep, 'end':end, 'file':file}
    """
    printargs = [arg for arg in gen_printargs(args)]
    mapping = {}
    pos = []
    for arg in printargs:
        if arg.type == syms.argument:
            kids = arg.children
            assert kids[0].type == token.NAME, repr(arg)
            assert len(kids) > 1, repr(arg)
            assert str(kids[0].value) in ("sep", "end", "file")
            assert str(kids[0].value) not in mapping, mapping
            mapping[str(kids[0].value)] = kids[2]
        elif arg.type == token.STAR:
            return (None, None)
        else:
            pos.append(arg)
    return (pos, mapping)

class FixPrint(fixer_base.BaseFix):

    PATTERN = """
              power< 'print' parens=trailer < '(' args=any* ')' > any* >
              """

    def match(self, node):
        """
        Since the tree needs to be fixed once and only once if and only if it
        matches, then we can start discarding matches after we make the first.
        """
        return super(FixPrint,self).match(node)

    def transform(self, node, results):
        args = results.get("args")
        if not args:
            parens = results.get("parens")
            parens.remove()
            return
        pos, opts = map_printargs(args)
        if pos is None or opts is None:
            self.cannot_convert(node, "-fprint does not support argument unpacking.  fix using -xprint and then again with  -fprintfunction.")
            return
        if "file" in opts and \
           "end" in opts and \
           opts["file"].type != token.NAME:
            self.warning(opts["file"], "file is not a variable name; "\
                   "print fixer suggests to bind the file to a variable "\
                   "name first before passing it to print function")
        try:
            with warnings.catch_warnings(record=True) as w:
                new_node = replace_print(pos, opts, old_node=node)
                if len(w) > 0:
                    self.warning(node, "coercing to unicode even though this may be a bytes object")
        except AttributeError:
            # Python 2.5 doesn't have warnings.catch_warnings, so we're in Python 2.5 code here...
            new_node = replace_print(pos, dict([(bytes(k), opts[k]) for k in opts]), old_node=node)

        new_node.prefix = node.prefix

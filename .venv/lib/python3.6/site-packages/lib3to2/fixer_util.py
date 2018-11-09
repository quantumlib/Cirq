from lib2to3.pygram import token, python_symbols as syms
from lib2to3.pytree import Leaf, Node
from lib2to3.fixer_util import *

def Star(prefix=None):
    return Leaf(token.STAR, '*', prefix=prefix)

def DoubleStar(prefix=None):
    return Leaf(token.DOUBLESTAR, '**', prefix=prefix)

def Minus(prefix=None):
    return Leaf(token.MINUS, '-', prefix=prefix)

def commatize(leafs):
    """
    Accepts/turns: (Name, Name, ..., Name, Name) 
    Returns/into: (Name, Comma, Name, Comma, ..., Name, Comma, Name)
    """
    new_leafs = []
    for leaf in leafs:
        new_leafs.append(leaf)
        new_leafs.append(Comma())
    del new_leafs[-1]
    return new_leafs

def indentation(node):
    """
    Returns the indentation for this node
    Iff a node is in a suite, then it has indentation.
    """
    while node.parent is not None and node.parent.type != syms.suite:
        node = node.parent
    if node.parent is None:
        return ""
    # The first three children of a suite are NEWLINE, INDENT, (some other node)
    # INDENT.value contains the indentation for this suite
    # anything after (some other node) has the indentation as its prefix.
    if node.type == token.INDENT:
        return node.value
    elif node.prev_sibling is not None and node.prev_sibling.type == token.INDENT:
        return node.prev_sibling.value
    elif node.prev_sibling is None:
        return ""
    else:
        return node.prefix

def indentation_step(node):
    """
    Dirty little trick to get the difference between each indentation level
    Implemented by finding the shortest indentation string
    (technically, the "least" of all of the indentation strings, but
    tabs and spaces mixed won't get this far, so those are synonymous.)
    """
    r = find_root(node)
    # Collect all indentations into one set.
    all_indents = set(i.value for i in r.pre_order() if i.type == token.INDENT)
    if not all_indents:
        # nothing is indented anywhere, so we get to pick what we want
        return "    " # four spaces is a popular convention
    else:
        return min(all_indents)

def suitify(parent):
    """
    Turn the stuff after the first colon in parent's children
    into a suite, if it wasn't already
    """
    for node in parent.children:
        if node.type == syms.suite:
            # already in the prefered format, do nothing
            return

    # One-liners have no suite node, we have to fake one up
    for i, node in enumerate(parent.children):
        if node.type == token.COLON:
            break
    else:
        raise ValueError("No class suite and no ':'!")
    # Move everything into a suite node
    suite = Node(syms.suite, [Newline(), Leaf(token.INDENT, indentation(node) + indentation_step(node))])
    one_node = parent.children[i+1]
    one_node.remove()
    one_node.prefix = ''
    suite.append_child(one_node)
    parent.append_child(suite)

def NameImport(package, as_name=None, prefix=None):
    """
    Accepts a package (Name node), name to import it as (string), and
    optional prefix and returns a node:
    import <package> [as <as_name>]
    """
    if prefix is None:
        prefix = ""
    children = [Name("import", prefix=prefix), package]
    if as_name is not None:
        children.extend([Name("as", prefix=" "),
                         Name(as_name, prefix=" ")])
    return Node(syms.import_name, children)

_compound_stmts = (syms.if_stmt, syms.while_stmt, syms.for_stmt, syms.try_stmt, syms.with_stmt)
_import_stmts = (syms.import_name, syms.import_from)
def import_binding_scope(node):
    """
    Generator yields all nodes for which a node (an import_stmt) has scope
    The purpose of this is for a call to _find() on each of them
    """
    # import_name / import_from are small_stmts
    assert node.type in _import_stmts
    test = node.next_sibling
    # A small_stmt can only be followed by a SEMI or a NEWLINE.
    while test.type == token.SEMI:
        nxt = test.next_sibling
        # A SEMI can only be followed by a small_stmt or a NEWLINE
        if nxt.type == token.NEWLINE:
            break
        else:
            yield nxt
        # A small_stmt can only be followed by either a SEMI or a NEWLINE
        test = nxt.next_sibling
    # Covered all subsequent small_stmts after the import_stmt
    # Now to cover all subsequent stmts after the parent simple_stmt
    parent = node.parent
    assert parent.type == syms.simple_stmt
    test = parent.next_sibling
    while test is not None:
        # Yes, this will yield NEWLINE and DEDENT.  Deal with it.
        yield test
        test = test.next_sibling

    context = parent.parent
    # Recursively yield nodes following imports inside of a if/while/for/try/with statement
    if context.type in _compound_stmts:
        # import is in a one-liner
        c = context
        while c.next_sibling is not None:
            yield c.next_sibling
            c = c.next_sibling
        context = context.parent

    # Can't chain one-liners on one line, so that takes care of that.

    p = context.parent
    if p is None:
        return

    # in a multi-line suite

    while p.type in _compound_stmts:

        if context.type == syms.suite:
            yield context

        context = context.next_sibling

        if context is None:
            context = p.parent
            p = context.parent
            if p is None:
                break

def ImportAsName(name, as_name, prefix=None):
    new_name = Name(name)
    new_as = Name("as", prefix=" ")
    new_as_name = Name(as_name, prefix=" ")
    new_node = Node(syms.import_as_name, [new_name, new_as, new_as_name])
    if prefix is not None:
        new_node.prefix = prefix
    return new_node

def future_import(feature, node):
    root = find_root(node)
    
    if does_tree_import("__future__", feature, node):
        return

    insert_pos = 0
    for idx, node in enumerate(root.children):
        if node.type == syms.simple_stmt and node.children and \
           node.children[0].type == token.STRING:
            insert_pos = idx + 1
            break

    for thing_after in root.children[insert_pos:]:
        if thing_after.type == token.NEWLINE:
            insert_pos += 1
            continue

        prefix = thing_after.prefix
        thing_after.prefix = ""
        break
    else:
        prefix = ""

    import_ = FromImport("__future__", [Leaf(token.NAME, feature, prefix=" ")])

    children = [import_, Newline()]
    root.insert_child(insert_pos, Node(syms.simple_stmt, children, prefix=prefix))

def parse_args(arglist, scheme):
    """
    Parse a list of arguments into a dict
    """
    arglist = [i for i in arglist if i.type != token.COMMA]
    
    ret_mapping = dict([(k, None) for k in scheme])

    for i, arg in enumerate(arglist):
        if arg.type == syms.argument and arg.children[1].type == token.EQUAL:
            # argument < NAME '=' any >
            slot = arg.children[0].value
            ret_mapping[slot] = arg.children[2]
        else:
            slot = scheme[i]
            ret_mapping[slot] = arg

    return ret_mapping

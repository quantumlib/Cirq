"""
Fixer for:
range(s) -> xrange(s)
list(range(s)) -> range(s)
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, is_probably_builtin
from lib2to3.pygram import python_symbols as syms
import token

def list_called(node):
    """
    Returns the power node that contains list as its first child if node
    is contained in a list() call, otherwise False.
    """
    parent = node.parent
    if parent is not None and parent.type == syms.trailer:
        prev = parent.prev_sibling
        if prev is not None and \
           prev.type == token.NAME and \
           prev.value == "list" and \
           is_probably_builtin(prev):
            return prev.parent
    return False

class FixRange(fixer_base.BaseFix):

    PATTERN = """
              power< name='range' trailer< '(' any ')' > >
              """

    def transform(self, node, results):
        name = results["name"]
        if not is_probably_builtin(name):
            return
        list_call = list_called(node)
        if list_call:
            new_node = node.clone()
            new_node.prefix = list_call.prefix
            parent = list_call.parent
            i = list_call.remove()
            for after in list_call.children[2:]:
                new_node.append_child(after)
            parent.insert_child(i, new_node)
        else:
            name.replace(Name("xrange", prefix=name.prefix))

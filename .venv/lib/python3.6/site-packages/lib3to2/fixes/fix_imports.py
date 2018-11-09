"""
Fixer for standard library imports renamed in Python 3
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, is_probably_builtin, Newline, does_tree_import
from lib2to3.pygram import python_symbols as syms
from lib2to3.pgen2 import token
from lib2to3.pytree import Node, Leaf

from ..fixer_util import NameImport

# used in simple_mapping_to_pattern()
MAPPING = {"reprlib": "repr",
           "winreg": "_winreg",
           "configparser": "ConfigParser",
           "copyreg": "copy_reg",
           "queue": "Queue",
           "socketserver": "SocketServer",
           "_markupbase": "markupbase",
           "test.support": "test.test_support",
           "dbm.bsd": "dbhash",
           "dbm.ndbm": "dbm",
           "dbm.dumb": "dumbdbm",
           "dbm.gnu": "gdbm",
           "html.parser": "HTMLParser",
           "html.entities": "htmlentitydefs",
           "http.client": "httplib",
           "http.cookies": "Cookie",
           "http.cookiejar": "cookielib",
           "tkinter": "Tkinter",
           "tkinter.dialog": "Dialog",
           "tkinter._fix": "FixTk",
           "tkinter.scrolledtext": "ScrolledText",
           "tkinter.tix": "Tix",
           "tkinter.constants": "Tkconstants",
           "tkinter.dnd": "Tkdnd",
           "tkinter.__init__": "Tkinter",
           "tkinter.colorchooser": "tkColorChooser",
           "tkinter.commondialog": "tkCommonDialog",
           "tkinter.font": "tkFont",
           "tkinter.messagebox": "tkMessageBox",
           "tkinter.turtle": "turtle",
           "urllib.robotparser": "robotparser",
           "xmlrpc.client": "xmlrpclib",
           "builtins": "__builtin__",
}

# generic strings to help build patterns
# these variables mean (with http.client.HTTPConnection as an example):
# name = http
# attr = client
# used = HTTPConnection
# fmt_name is a formatted subpattern (simple_name_match or dotted_name_match)

# helps match 'queue', as in 'from queue import ...'
simple_name_match = "name='{name}'"
# helps match 'client', to be used if client has been imported from http
subname_match = "attr='{attr}'"
# helps match 'http.client', as in 'import urllib.request'
dotted_name_match = "dotted_name=dotted_name< {fmt_name} '.' {fmt_attr} >"
# helps match 'queue', as in 'queue.Queue(...)'
power_onename_match = "{fmt_name}"
# helps match 'http.client', as in 'http.client.HTTPConnection(...)'
power_twoname_match = "power< {fmt_name} trailer< '.' {fmt_attr} > any* >"
# helps match 'client.HTTPConnection', if 'client' has been imported from http
power_subname_match = "power< {fmt_attr} any* >"
# helps match 'from http.client import HTTPConnection'
from_import_match = "from_import=import_from< 'from' {fmt_name} 'import' ['('] imported=any [')'] >"
# helps match 'from http import client'
from_import_submod_match = "from_import_submod=import_from< 'from' {fmt_name} 'import' ({fmt_attr} | import_as_name< {fmt_attr} 'as' renamed=any > | import_as_names< any* ({fmt_attr} | import_as_name< {fmt_attr} 'as' renamed=any >) any* > ) >"
# helps match 'import urllib.request'
name_import_match = "name_import=import_name< 'import' {fmt_name} > | name_import=import_name< 'import' dotted_as_name< {fmt_name} 'as' renamed=any > >"
# helps match 'import http.client, winreg'
multiple_name_import_match = "name_import=import_name< 'import' dotted_as_names< names=any* > >"

def all_patterns(name):
    """
    Accepts a string and returns a pattern of possible patterns involving that name
    Called by simple_mapping_to_pattern for each name in the mapping it receives.
    """

    # i_ denotes an import-like node
    # u_ denotes a node that appears to be a usage of the name
    if '.' in name:
        name, attr = name.split('.', 1)
        simple_name = simple_name_match.format(name=name)
        simple_attr = subname_match.format(attr=attr)
        dotted_name = dotted_name_match.format(fmt_name=simple_name, fmt_attr=simple_attr)
        i_from = from_import_match.format(fmt_name=dotted_name)
        i_from_submod = from_import_submod_match.format(fmt_name=simple_name, fmt_attr=simple_attr)
        i_name = name_import_match.format(fmt_name=dotted_name)
        u_name = power_twoname_match.format(fmt_name=simple_name, fmt_attr=simple_attr)
        u_subname = power_subname_match.format(fmt_attr=simple_attr)
        return ' | \n'.join((i_name, i_from, i_from_submod, u_name, u_subname))
    else:
        simple_name = simple_name_match.format(name=name)
        i_name = name_import_match.format(fmt_name=simple_name)
        i_from = from_import_match.format(fmt_name=simple_name)
        u_name = power_onename_match.format(fmt_name=simple_name)
        return ' | \n'.join((i_name, i_from, u_name))


class FixImports(fixer_base.BaseFix):

    order = "pre"

    PATTERN = ' | \n'.join([all_patterns(name) for name in MAPPING])
    PATTERN = ' | \n'.join((PATTERN, multiple_name_import_match))

    def fix_dotted_name(self, node, mapping=MAPPING):
        """
        Accepts either a DottedName node or a power node with a trailer.
        If mapping is given, use it; otherwise use our MAPPING
        Returns a node that can be in-place replaced by the node given
        """
        if node.type == syms.dotted_name:
            _name = node.children[0]
            _attr = node.children[2]
        elif node.type == syms.power:
            _name = node.children[0]
            _attr = node.children[1].children[1]
        name = _name.value
        attr = _attr.value
        full_name = name + '.' + attr
        if not full_name in mapping:
            return
        to_repl = mapping[full_name]
        if '.' in to_repl:
            repl_name, repl_attr = to_repl.split('.')
            _name.replace(Name(repl_name, prefix=_name.prefix))
            _attr.replace(Name(repl_attr, prefix=_attr.prefix))
        elif node.type == syms.dotted_name:
            node.replace(Name(to_repl, prefix=node.prefix))
        elif node.type == syms.power:
            _name.replace(Name(to_repl, prefix=_name.prefix))
            parent = _attr.parent
            _attr.remove()
            parent.remove()

    def fix_simple_name(self, node, mapping=MAPPING):
        """
        Accepts a Name leaf.
        If mapping is given, use it; otherwise use our MAPPING
        Returns a node that can be in-place replaced by the node given
        """
        assert node.type == token.NAME, repr(node)
        if not node.value in mapping:
            return
        replacement = mapping[node.value]
        node.replace(Leaf(token.NAME, str(replacement), prefix=node.prefix))

    def fix_submod_import(self, imported, name, node):
        """
        Accepts a list of NAME leafs, a name string, and a node
        node is given as an argument to BaseFix.transform()
        NAME leafs come from an import_as_names node (the children)
        name string is the base name found in node.
        """
        submods = []
        missed = []
        for attr in imported:
            dotted = '.'.join((name, attr.value))
            if dotted in MAPPING:
                # get the replacement module
                to_repl = MAPPING[dotted]
                if '.' not in to_repl:
                    # it's a simple name, so use a simple replacement.
                    _import = NameImport(Name(to_repl, prefix=" "), attr.value)
                    submods.append(_import)
            elif attr.type == token.NAME:
                missed.append(attr.clone())
        if not submods:
            return

        parent = node.parent
        node.replace(submods[0])
        if len(submods) > 1:
            start = submods.pop(0)
            prev = start
            for submod in submods:
                parent.append_child(submod)
        if missed:
            self.warning(node, "Imported names not known to 3to2 to be part of the package {0}.  Leaving those alone... high probability that this code will be incorrect.".format(name))
            children = [Name("from"), Name(name, prefix=" "), Name("import", prefix=" "), Node(syms.import_as_names, missed)]
            orig_stripped = Node(syms.import_from, children)
            parent.append_child(Newline())
            parent.append_child(orig_stripped)


    def get_dotted_import_replacement(self, name_node, attr_node, mapping=MAPPING, renamed=None):
        """
        For (http, client) given and httplib being the correct replacement,
        returns (httplib as client, None)
        For (test, support) given and test.test_support being the replacement,
        returns (test, test_support as support)
        """
        full_name = name_node.value + '.' + attr_node.value
        replacement = mapping[full_name]
        if '.' in replacement:
            new_name, new_attr = replacement.split('.')
            if renamed is None:
                return Name(new_name, prefix=name_node.prefix), Node(syms.dotted_as_name, [Name(new_attr, prefix=attr_node.prefix), Name('as', prefix=" "), attr_node.clone()])
            else:
                return Name(new_name, prefix=name_node.prefix), Name(new_attr, prefix=attr_node.prefix)
        else:
            return Node(syms.dotted_as_name, [Name(replacement, prefix=name_node.prefix), Name('as', prefix=' '), Name(attr_node.value, prefix=attr_node.prefix)]), None
    
    def transform(self, node, results):
        from_import = results.get("from_import")
        from_import_submod = results.get("from_import_submod")
        name_import = results.get("name_import")
        dotted_name = results.get("dotted_name")
        name = results.get("name")
        names = results.get("names")
        attr = results.get("attr")
        imported = results.get("imported")
        if names:
            for name in names:
                if name.type == token.NAME:
                    self.fix_simple_name(name)
                elif name.type == syms.dotted_as_name:
                    self.fix_simple_name(name.children[0]) if name.children[0].type == token.NAME else \
                    self.fix_dotted_name(name.children[0])
                elif name.type == syms.dotted_name:
                    self.fix_dotted_name(name)
        elif from_import_submod:
            renamed = results.get("renamed")
            new_name, new_attr = self.get_dotted_import_replacement(name, attr, renamed=renamed)
            if new_attr is not None:
                name.replace(new_name)
                attr.replace(new_attr)
            else:
                children = [Name("import"), new_name]
                node.replace(Node(syms.import_name, children, prefix=node.prefix))
        elif dotted_name:
            self.fix_dotted_name(dotted_name)
        elif name_import or from_import:
            self.fix_simple_name(name)
        elif name and not attr:
            if does_tree_import(None, MAPPING[name.value], node) and \
               is_probably_builtin(name):
                self.fix_simple_name(name)
        elif name and attr:
            # Note that this will fix a dotted name that was never imported.  This will probably not matter.
            self.fix_dotted_name(node)
        elif imported and imported.type == syms.import_as_names:
            self.fix_submod_import(imported=imported.children, node=node, name=name.value)

"""
Fixer for complicated imports
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, String, FromImport, Newline, Comma
from ..fixer_util import token, syms, Leaf, Node, Star, indentation, ImportAsName

TK_BASE_NAMES = ('ACTIVE', 'ALL', 'ANCHOR', 'ARC','BASELINE', 'BEVEL', 'BOTH',
                 'BOTTOM', 'BROWSE', 'BUTT', 'CASCADE', 'CENTER', 'CHAR',
                 'CHECKBUTTON', 'CHORD', 'COMMAND', 'CURRENT', 'DISABLED',
                 'DOTBOX', 'E', 'END', 'EW', 'EXCEPTION', 'EXTENDED', 'FALSE',
                 'FIRST', 'FLAT', 'GROOVE', 'HIDDEN', 'HORIZONTAL', 'INSERT',
                 'INSIDE', 'LAST', 'LEFT', 'MITER', 'MOVETO', 'MULTIPLE', 'N',
                 'NE', 'NO', 'NONE', 'NORMAL', 'NS', 'NSEW', 'NUMERIC', 'NW',
                 'OFF', 'ON', 'OUTSIDE', 'PAGES', 'PIESLICE', 'PROJECTING',
                 'RADIOBUTTON', 'RAISED', 'READABLE', 'RIDGE', 'RIGHT',
                 'ROUND', 'S', 'SCROLL', 'SE', 'SEL', 'SEL_FIRST', 'SEL_LAST',
                 'SEPARATOR', 'SINGLE', 'SOLID', 'SUNKEN', 'SW', 'StringTypes',
                 'TOP', 'TRUE', 'TclVersion', 'TkVersion', 'UNDERLINE', 
                 'UNITS', 'VERTICAL', 'W', 'WORD', 'WRITABLE', 'X', 'Y', 'YES',
                 'wantobjects')

PY2MODULES = { 
              'urllib2' : (
                  'AbstractBasicAuthHandler', 'AbstractDigestAuthHandler',
                  'AbstractHTTPHandler', 'BaseHandler', 'CacheFTPHandler',
                  'FTPHandler', 'FileHandler', 'HTTPBasicAuthHandler',
                  'HTTPCookieProcessor', 'HTTPDefaultErrorHandler',
                  'HTTPDigestAuthHandler', 'HTTPError', 'HTTPErrorProcessor',
                  'HTTPHandler', 'HTTPPasswordMgr',
                  'HTTPPasswordMgrWithDefaultRealm', 'HTTPRedirectHandler',
                  'HTTPSHandler', 'OpenerDirector', 'ProxyBasicAuthHandler',
                  'ProxyDigestAuthHandler', 'ProxyHandler', 'Request',
                  'StringIO', 'URLError', 'UnknownHandler', 'addinfourl',
                  'build_opener', 'install_opener', 'parse_http_list',
                  'parse_keqv_list', 'randombytes', 'request_host', 'urlopen'),
              'urllib' : (
                  'ContentTooShortError', 'FancyURLopener','URLopener',
                  'basejoin', 'ftperrors', 'getproxies',
                  'getproxies_environment', 'localhost', 'pathname2url',
                  'quote', 'quote_plus', 'splitattr', 'splithost',
                  'splitnport', 'splitpasswd', 'splitport', 'splitquery',
                  'splittag', 'splittype', 'splituser', 'splitvalue',
                  'thishost', 'unquote', 'unquote_plus', 'unwrap',
                  'url2pathname', 'urlcleanup', 'urlencode', 'urlopen',
                  'urlretrieve',),
              'urlparse' : (
                  'parse_qs', 'parse_qsl', 'urldefrag', 'urljoin',
                  'urlparse', 'urlsplit', 'urlunparse', 'urlunsplit'),
              'dbm' : (
                  'ndbm', 'gnu', 'dumb'),
              'anydbm' : (
                  'error', 'open'),
              'whichdb' : (
                  'whichdb',),
              'BaseHTTPServer' : (
                  'BaseHTTPRequestHandler', 'HTTPServer'),
              'CGIHTTPServer' : (
                  'CGIHTTPRequestHandler',),
              'SimpleHTTPServer' : (
                  'SimpleHTTPRequestHandler',),
              'FileDialog' : TK_BASE_NAMES + (
                  'FileDialog', 'LoadFileDialog', 'SaveFileDialog',
                  'dialogstates', 'test'),
              'tkFileDialog' : (
                  'Directory', 'Open', 'SaveAs', '_Dialog', 'askdirectory',
                  'askopenfile', 'askopenfilename', 'askopenfilenames',
                  'askopenfiles', 'asksaveasfile', 'asksaveasfilename'),
              'SimpleDialog' : TK_BASE_NAMES + (
                  'SimpleDialog',),
              'tkSimpleDialog' : TK_BASE_NAMES + (
                  'askfloat', 'askinteger', 'askstring', 'Dialog'),
              'SimpleXMLRPCServer' : (
                  'CGIXMLRPCRequestHandler', 'SimpleXMLRPCDispatcher',
                  'SimpleXMLRPCRequestHandler', 'SimpleXMLRPCServer',
                  'list_public_methods', 'remove_duplicates',
                  'resolve_dotted_attribute'),
              'DocXMLRPCServer' : (
                  'DocCGIXMLRPCRequestHandler', 'DocXMLRPCRequestHandler',
                  'DocXMLRPCServer', 'ServerHTMLDoc','XMLRPCDocGenerator'),
                }

MAPPING = { 'urllib.request' :
                ('urllib2', 'urllib'),
            'urllib.error' :
                ('urllib2', 'urllib'),
            'urllib.parse' :
                ('urllib2', 'urllib', 'urlparse'),
            'dbm.__init__' :
                ('anydbm', 'whichdb'),
            'http.server' :
                ('CGIHTTPServer', 'SimpleHTTPServer', 'BaseHTTPServer'),
            'tkinter.filedialog' :
                ('tkFileDialog', 'FileDialog'),
            'tkinter.simpledialog' :
                ('tkSimpleDialog', 'SimpleDialog'),
            'xmlrpc.server' :
                ('DocXMLRPCServer', 'SimpleXMLRPCServer'),
            }

# helps match 'http', as in 'from http.server import ...'
simple_name = "name='%s'"
# helps match 'server', as in 'from http.server import ...'
simple_attr = "attr='%s'"
# helps match 'HTTPServer', as in 'from http.server import HTTPServer'
simple_using = "using='%s'"
# helps match 'urllib.request', as in 'import urllib.request'
dotted_name = "dotted_name=dotted_name< %s '.' %s >"
# helps match 'http.server', as in 'http.server.HTTPServer(...)'
power_twoname = "pow=power< %s trailer< '.' %s > trailer< '.' using=any > any* >"
# helps match 'dbm.whichdb', as in 'dbm.whichdb(...)'
power_onename = "pow=power< %s trailer< '.' using=any > any* >"
# helps match 'from http.server import HTTPServer'
# also helps match 'from http.server import HTTPServer, SimpleHTTPRequestHandler'
# also helps match 'from http.server import *'
from_import = "from_import=import_from< 'from' %s 'import' (import_as_name< using=any 'as' renamed=any> | in_list=import_as_names< using=any* > | using='*' | using=NAME) >"
# helps match 'import urllib.request'
name_import = "name_import=import_name< 'import' (%s | in_list=dotted_as_names< imp_list=any* >) >"

#############
# WON'T FIX #
#############

# helps match 'import urllib.request as name'
name_import_rename = "name_import_rename=dotted_as_name< %s 'as' renamed=any >"
# helps match 'from http import server'
from_import_rename = "from_import_rename=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | in_list=import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* >) >"

def all_modules_subpattern():
    """
    Builds a pattern for all toplevel names
    (urllib, http, etc)
    """
    names_dot_attrs = [mod.split(".") for mod in MAPPING]
    ret = "( " + " | ".join([dotted_name % (simple_name % (mod[0]),
                                            simple_attr % (mod[1])) for mod in names_dot_attrs])
    ret += " | "
    ret += " | ".join([simple_name % (mod[0]) for mod in names_dot_attrs if mod[1] == "__init__"]) + " )"
    return ret

def all_candidates(name, attr, MAPPING=MAPPING):
    """
    Returns all candidate packages for the name.attr
    """
    dotted = name + '.' + attr
    assert dotted in MAPPING, "No matching package found."
    ret = MAPPING[dotted]
    if attr == '__init__':
        return ret + (name,)
    return ret

def new_package(name, attr, using, MAPPING=MAPPING, PY2MODULES=PY2MODULES):
    """
    Returns which candidate package for name.attr provides using
    """
    for candidate in all_candidates(name, attr, MAPPING):
        if using in PY2MODULES[candidate]:
            break
    else:
        candidate = None

    return candidate

def build_import_pattern(mapping1, mapping2):
    """
    mapping1: A dict mapping py3k modules to all possible py2k replacements
    mapping2: A dict mapping py2k modules to the things they do
    This builds a HUGE pattern to match all ways that things can be imported
    """
    # py3k: urllib.request, py2k: ('urllib2', 'urllib')
    yield from_import % (all_modules_subpattern())
    for py3k, py2k in mapping1.items():
        name, attr = py3k.split('.')
        s_name = simple_name % (name)
        s_attr = simple_attr % (attr)
        d_name = dotted_name % (s_name, s_attr)
        yield name_import % (d_name)
        yield power_twoname % (s_name, s_attr)
        if attr == '__init__':
            yield name_import % (s_name)
            yield power_onename % (s_name)
        yield name_import_rename % (d_name)
        yield from_import_rename % (s_name, s_attr, s_attr, s_attr, s_attr)

def name_import_replacement(name, attr):
    children = [Name("import")]
    for c in all_candidates(name.value, attr.value):
        children.append(Name(c, prefix=" "))
        children.append(Comma())
    children.pop()
    replacement = Node(syms.import_name, children)
    return replacement


class FixImports2(fixer_base.BaseFix):

    run_order = 4

    PATTERN = " | \n".join(build_import_pattern(MAPPING, PY2MODULES))

    def transform(self, node, results):
        # The patterns dictate which of these names will be defined
        name = results.get("name")
        attr = results.get("attr")
        if attr is None:
            attr = Name("__init__")
        using = results.get("using")
        in_list = results.get("in_list")
        imp_list = results.get("imp_list")
        power = results.get("pow")
        before = results.get("before")
        after = results.get("after")
        d_name = results.get("dotted_name")
        # An import_stmt is always contained within a simple_stmt
        simple_stmt = node.parent
        # The parent is useful for adding new import_stmts
        parent = simple_stmt.parent
        idx = parent.children.index(simple_stmt)
        if any((results.get("from_import_rename") is not None,
                results.get("name_import_rename") is not None)): 
            self.cannot_convert(node, reason="ambiguity: import binds a single name")

        elif using is None and not in_list:
            # import urllib.request, single-name import
            replacement = name_import_replacement(name, attr)
            replacement.prefix = node.prefix
            node.replace(replacement)

        elif using is None:
            # import ..., urllib.request, math, http.sever, ...
            for d_name in imp_list:
                if d_name.type == syms.dotted_name:
                    name = d_name.children[0]
                    attr = d_name.children[2]
                elif d_name.type == token.NAME and d_name.value + ".__init__" in MAPPING:
                    name = d_name
                    attr = Name("__init__")
                else:
                    continue
                if name.value + "." + attr.value not in MAPPING:
                    continue
                candidates = all_candidates(name.value, attr.value)
                children = [Name("import")]
                for c in candidates:
                    children.append(Name(c, prefix=" "))
                    children.append(Comma())
                children.pop()
                # Put in the new statement.
                indent = indentation(simple_stmt)
                next_stmt = Node(syms.simple_stmt, [Node(syms.import_name, children), Newline()])
                parent.insert_child(idx+1, next_stmt)
                parent.insert_child(idx+1, Leaf(token.INDENT, indent))
                # Remove the old imported name
                test_comma = d_name.next_sibling
                if test_comma and test_comma.type == token.COMMA:
                    test_comma.remove()
                elif test_comma is None:
                    test_comma = d_name.prev_sibling
                    if test_comma and test_comma.type == token.COMMA:
                        test_comma.remove()
                d_name.remove()
            if not in_list.children:
                simple_stmt.remove()

        elif in_list is not None:
            ##########################################################
            # "from urllib.request import urlopen, urlretrieve, ..." #
            # Replace one import statement with potentially many.    #
            ##########################################################
            packages = dict([(n,[]) for n in all_candidates(name.value,
                                                            attr.value)])
            # Figure out what names need to be imported from what
            # Add them to a dict to be parsed once we're completely done
            for imported in using:
                if imported.type == token.COMMA:
                    continue
                if imported.type == syms.import_as_name:
                    test_name = imported.children[0].value
                    if len(imported.children) > 2:
                        # 'as' whatever
                        rename = imported.children[2].value
                    else:
                        rename = None
                elif imported.type == token.NAME:
                    test_name = imported.value
                    rename = None
                pkg = new_package(name.value, attr.value, test_name)
                packages[pkg].append((test_name, rename))

            # Parse the dict to create new import statements to replace this one
            imports = []
            for new_pkg, names in packages.items():
                if not names:
                    # Didn't import anything from that package, move along
                    continue
                new_names = []
                for test_name, rename in names:
                    if rename is None:
                        new_names.append(Name(test_name, prefix=" "))
                    else:
                        new_names.append(ImportAsName(test_name, rename, prefix=" "))
                    new_names.append(Comma())
                new_names.pop()
                imports.append(FromImport(new_pkg, new_names))
            # Replace this import statement with one of the others
            replacement = imports.pop()
            replacement.prefix = node.prefix
            node.replace(replacement)
            indent = indentation(simple_stmt)
            # Add the remainder of the imports as new statements.
            while imports:
                next_stmt = Node(syms.simple_stmt, [imports.pop(), Newline()])
                parent.insert_child(idx+1, next_stmt)
                parent.insert_child(idx+1, Leaf(token.INDENT, indent))

        elif using.type == token.STAR:
            # from urllib.request import *
            nodes = [FromImport(pkg, [Star(prefix=" ")]) for pkg in
                                        all_candidates(name.value, attr.value)]
            replacement = nodes.pop()
            replacement.prefix = node.prefix
            node.replace(replacement)
            indent = indentation(simple_stmt)
            while nodes:
                next_stmt = Node(syms.simple_stmt, [nodes.pop(), Newline()])
                parent.insert_child(idx+1, next_stmt)
                parent.insert_child(idx+1, Leaf(token.INDENT, indent))
        elif power is not None:
            # urllib.request.urlopen
            # Replace it with urllib2.urlopen
            pkg = new_package(name.value, attr.value, using.value)
            # Remove the trailer node that contains attr.
            if attr.parent:
                attr.parent.remove()
            name.replace(Name(pkg, prefix=name.prefix))

        elif using.type == token.NAME:
            # from urllib.request import urlopen
            pkg = new_package(name.value, attr.value, using.value)
            if attr.value == "__init__" and pkg == name.value:
                # Replacing "from abc import xyz" with "from abc import xyz"
                # Just leave it alone so as not to mess with other fixers
                return
            else:
                node.replace(FromImport(pkg, [using]))

"""
Fixer for:
collections.UserDict -> UserDict.UserDict
collections.UserList -> UserList.UserList
collections.UserString -> UserString.UserString
"""

from lib2to3 import fixer_base
from ..fixer_util import Name, syms, touch_import

class FixCollections(fixer_base.BaseFix):

    explicit = True

    PATTERN = """import_from< 'from' collections='collections' 'import' name=('UserDict' | 'UserList' | 'UserString') > |
                 power< collections='collections' trailer< '.' name=('UserDict' | 'UserList' | 'UserString') > any* >"""

    def transform(self, node, results):

        collections = results["collections"]
        name = results["name"][0]

        collections.replace(Name(name.value, prefix=collections.prefix))
        if node.type == syms.power:
            touch_import(None, name.value, node)

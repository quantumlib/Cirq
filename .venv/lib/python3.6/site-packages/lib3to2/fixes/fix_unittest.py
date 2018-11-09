"""
Fixer for unittest -> unittest2
"""

from lib2to3 import fixer_base
from ..fixer_util import Name

class FixUnittest(fixer_base.BaseFix):

    explicit = True

    PATTERN = """
    import_from< 'from' name='unittest' 'import' any > |
    import_name< 'import' (name='unittest' | dotted_as_name< name='unittest' 'as' any >) > |
    import_name< 'import' dotted_as_names< any* (name='unittest' | dotted_as_name< name='unittest' 'as' any >) any* > > |
    power< name='unittest' any* >"""

    def transform(self, node, results):
        name = results['name']
        name.replace(Name("unittest2", prefix=name.prefix))

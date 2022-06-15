"""temp docstring"""

import astroid
from pylint import checkers, interfaces
from pylint.checkers import utils


class ImportOnlyModulesChecker(checkers.BaseChecker):
    """A Checker to check for only module imports."""
    __implements__ = interfaces.IAstroidChecker

    name = 'import-only-modules'
    priority = -1
    msgs = {
        'W5521': (
            "Import \"%s\" from \"%s\" is not a module.",
            'import-only-modules',
            "Only modules should be imported.",
        ),
    }

    @utils.check_messages('import-only-modules')
    def visit_importfrom(self, node):
        """A visitor for import statements."""
        try:
            imported_module = node.do_import_module(node.modname)
        except astroid.AstroidBuildingException:
            # Import errors should be checked elsewhere.
            return

        if node.level is None:
            modname = node.modname
        else:
            modname = '.' * node.level + node.modname

        for (name, _) in node.names:
            # Wildcard imports should be checked elsewhere.
            if name == '*':
                continue

            try:
                imported_module.import_module(name, True)
                # Good, we could import "name" as a module relative to the "imported_module".
            except astroid.AstroidImportError:
                self.add_message(
                    'import-only-modules',
                    node=node,
                )
            except astroid.AstroidBuildingException:
                # Some other error.
                pass


def register(linter):
    """temp docstring"""
    linter.register_checker(ImportOnlyModulesChecker(linter))

"""
Fixer for os.getcwd() -> os.getcwdu().
Also warns about "from os import getcwd", suggesting the above form.
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name

class FixGetcwd(fixer_base.BaseFix):

    PATTERN = """
              power< 'os' trailer< dot='.' name='getcwd' > any* >
              |
              import_from< 'from' 'os' 'import' bad='getcwd' >
              """

    def transform(self, node, results):
        if "name" in results:
            name = results["name"]
            name.replace(Name("getcwdu", prefix=name.prefix))
        elif "bad" in results:
            # Can't convert to getcwdu and then expect to catch every use.
            self.cannot_convert(node, "import os, use os.getcwd() instead.")
            return
        else:
            raise ValueError("For some reason, the pattern matcher failed.")

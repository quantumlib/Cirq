# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:53:41 2013

@author: silvester
"""
import os
from distutils import log
from distutils.command.build_py import build_py
from lib2to3 import refactor
from lib2to3 import pygram


class DistutilsRefactoringTool(refactor.RefactoringTool):
    """Refactoring tool for lib3to2 building"""
    def __init__(self, fixers, options=None, explicit=None):
        super(DistutilsRefactoringTool, self).__init__(fixers, options, explicit)
        self.driver.grammar = pygram.python_grammar_no_print_statement

    def refactor_string(self, data, name):
        """Override to keep print statements out of the grammar"""
        try:
            tree = self.driver.parse_string(data)
        except Exception, err:
            self.log_error("Can't parse %s: %s: %s",
                           name, err.__class__.__name__, err)
            return
        self.log_debug("Refactoring %s", name)
        self.refactor_tree(tree, name)
        return tree
        
    def log_error(self, msg, *args, **kw):
        log.error(msg, *args)

    def log_message(self, msg, *args):
        log.info(msg, *args)
    
    def log_debug(self, msg, *args):
        log.debug(msg, *args)


def run_3to2(files, fixer_names=None, options=None, explicit=None):
    """Invoke 3to2 on a list of Python files.
    The files should all come from the build area, as the
    modification is done in-place. To reduce the build time,
    only files modified since the last invocation of this
    function should be passed in the files argument."""

    if not files:
        return
    if fixer_names is None:
        fixer_names = refactor.get_fixers_from_package('lib3to2.fixes')
    r = DistutilsRefactoringTool(fixer_names, options=options)
    r.refactor(files, write=True)
    
    
def copydir_run_3to2(src, dest, template=None, fixer_names=None,
                     options=None, explicit=None):
    """Recursively copy a directory, only copying new and changed files,
    running run_3to2 over all newly copied Python modules afterward.

    If you give a template string, it's parsed like a MANIFEST.in.
    """
    from distutils.dir_util import mkpath
    from distutils.file_util import copy_file
    from distutils.filelist import FileList
    filelist = FileList()
    curdir = os.getcwd()
    os.chdir(src)
    try:
        filelist.findall()
    finally:
        os.chdir(curdir)
    filelist.files[:] = filelist.allfiles
    if template:
        for line in template.splitlines():
            line = line.strip()
            if not line: continue
            filelist.process_template_line(line)
    copied = []
    for filename in filelist.files:
        outname = os.path.join(dest, filename)
        mkpath(os.path.dirname(outname))
        res = copy_file(os.path.join(src, filename), outname, update=1)
        if res[1]: copied.append(outname)
    run_3to2([fn for fn in copied if fn.lower().endswith('.py')],
             fixer_names=fixer_names, options=options, explicit=explicit)
    return copied


class Mixin3to2:
    '''Mixin class for commands that run 3to2.
    To configure 3to2, setup scripts may either change
    the class variables, or inherit from individual commands
    to override how 3to2 is invoked.'''

    # provide list of fixers to run;
    # defaults to all from lib3to2.fixers
    fixer_names = None

    # options dictionary
    options = None

    # list of fixers to invoke even though they are marked as explicit
    explicit = None

    def run_3to2(self, files):
        return run_3to2(files, self.fixer_names, self.options, self.explicit)
        

class build_py_3to2(build_py, Mixin3to2):
    def run(self):
        self.updated_files = []

        # Base class code
        if self.py_modules:
            self.build_modules()
        if self.packages:
            self.build_packages()
            self.build_package_data()

        # 3to2
        self.run_3to2(self.updated_files)

        # Remaining base class code
        self.byte_compile(self.get_outputs(include_bytecode=0))

    def build_module(self, module, module_file, package):
        res = build_py.build_module(self, module, module_file, package)
        if res[1]:
            # file was copied
            self.updated_files.append(res[0])
        return res
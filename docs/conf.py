# -*- coding: utf-8 -*-
# coverage: ignore

# Configuration file for the Sphinx documentation builder.
# See http://www.sphinx-doc.org/en/master/config for help

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import inspect
import re
from typing import List, Any

import os
import sys

import pypandoc

cirq_root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, cirq_root_path)
from cirq import _doc


def setup(app):
    app.add_config_value('pandoc_use_parser', 'markdown', True)
    app.connect('autodoc-process-docstring', autodoc_process)
    app.connect('autodoc-skip-member', autodoc_skip_member)


def convert_markdown_mathjax_for_rst(lines: List[str]) -> List[str]:
    if all('$$' not in line for line in lines):
        return lines

    data = '\n'.join(lines)
    sections = data.split('$$')
    if len(sections) % 2 != 1:
        raise ValueError('Mismatched number of "$$" latex tokens.')

    result = []
    for i, s in enumerate(sections):
        if i % 2:
            # Avoid getting split across divs.
            s = ' '.join(s.split('\n'))
            # Avoid intermediate layers turning our newlines into slashes.
            s = s.replace('\\\\', r'\newline')
            # Turn latex like "|x\rangle" into "|x \rangle".
            # The extra space seems to be necessary to survive a later pass.
            s = re.sub(r'([a-zA-Z0-9])\\', r'\1 \\', s)
            # Keep the $$ so MathJax can find it.
            result.append('$${}$$'.format(s))
        else:
            # Work around bad table detection in pandoc by concatenating
            # lines from the same paragraph.
            s = '\n\n'.join(e.replace('\n', ' ') for e in s.split('\n\n'))

            # Convert markdown to rst.
            out = pypandoc.convert(s, to='rst', format='markdown_github')

            # Not sure why pandoc is escaping these...
            out = out.replace(r'\|', '|')

            result.extend(out.split('\n'))

    return result


def autodoc_skip_member(
        app,
        what: str,
        name: str,
        obj: Any,
        skip: bool,
        options,
) -> bool:
    """Public members already kept. Also include members marked as documented.
    """
    return id(obj) not in _doc.RECORDED_CONST_DOCS


def autodoc_process(app, what: str, name: str, obj: Any, options,
                    lines: List[str]) -> None:
    # Try to lookup in documented dictionary.
    found = _doc.RECORDED_CONST_DOCS.get(id(obj))
    if name.startswith('cirq') and found is not None:
        # Override docstring if requested.
        if found.doc_string is not None:
            new_doc_string = inspect.cleandoc(found.doc_string)
            lines[:] = new_doc_string.split('\n')
    elif not (getattr(obj, '__module__', 'cirq') or '').startswith('cirq'):
        # Don't convert objects from other modules.
        return

    # Don't convert output from Napoleon extension, which is already rst.
    i = 0
    while i < len(lines) and not lines[i].startswith(':'):
        i += 1
    if not i:
        return

    converted_lines = convert_markdown_mathjax_for_rst(lines[:i])
    kept_lines = lines[i:]

    data = pypandoc.convert(
        '\n'.join(converted_lines),
        to='rst',
        format='markdown_github',
    )

    lines[:] = data.split('\n') + kept_lines


# -- Project information -----------------------------------------------------

project = 'Cirq'
copyright = '2018, The Cirq Developers'  # pylint: disable=redefined-builtin
author = 'The Cirq Developers'

# The full version, including alpha/beta/rc tags
__version__ = ''
exec(open(os.path.join(cirq_root_path, 'cirq', '_version.py')).read())
release = __version__

# The short X.Y version
version = release  # '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'recommonmark',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Allow markdown includes.
# http://www.sphinx-doc.org/en/master/markdown.html
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output ---------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_favicon = 'favicon.ico'
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

html_logo = '_static/Cirq_logo_notext.png'
html_css_files = ['tweak-style.css']


# -- Options for HTMLHelp output -----------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'Cirqdoc'


# -- Options for LaTeX output --------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',

    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Cirq.tex', 'Cirq Documentation',
     'The Cirq Developers', 'manual'),
]


# -- Options for manual page output --------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'cirq', 'Cirq Documentation',
     [author], 1)
]


# -- Options for Texinfo output ------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Cirq', 'Cirq Documentation',
     author, 'Cirq', 'A python library for NISQ circuits.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------

# Generate subpages for reference docs automatically.
# http://www.sphinx-doc.org/en/master/ext/autosummary.html#generating-stub-pages-automatically
autosummary_generate = True

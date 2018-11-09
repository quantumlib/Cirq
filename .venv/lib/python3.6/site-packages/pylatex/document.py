# -*- coding: utf-8 -*-
"""
This module implements the class that deals with the full document.

..  :copyright: (c) 2014 by Jelte Fennema.
    :license: MIT, see License for more details.
"""

import os
import subprocess
import errno
from .base_classes import Environment, Command, Container, LatexObject, \
    UnsafeCommand
from .package import Package
from .errors import CompilerError
from .utils import dumps_list, rm_temp_dir, NoEscape
import pylatex.config as cf


class Document(Environment):
    r"""
    A class that contains a full LaTeX document.

    If needed, you can append stuff to the preamble or the packages.
    For instance, if you need to use ``\maketitle`` you can add the title,
    author and date commands to the preamble to make it work.

    """

    def __init__(self, default_filepath='default_filepath', *,
                 documentclass='article', document_options=None, fontenc='T1',
                 inputenc='utf8', font_size="normalsize", lmodern=True,
                 textcomp=True, microtype=None, page_numbers=True, indent=None,
                 geometry_options=None, data=None):
        r"""
        Args
        ----
        default_filepath: str
            The default path to save files.
        documentclass: str or `~.Command`
            The LaTeX class of the document.
        document_options: str or `list`
            The options to supply to the documentclass
        fontenc: str
            The option for the fontenc package. If it is `None`, the fontenc
            package will not be loaded at all.
        inputenc: str
            The option for the inputenc package. If it is `None`, the inputenc
            package will not be loaded at all.
        font_size: str
            The font size to declare as normalsize
        lmodern: bool
            Use the Latin Modern font. This is a font that contains more glyphs
            than the standard LaTeX font.
        textcomp: bool
            Adds even more glyphs, for instance the Euro (€) sign.
        page_numbers: bool
            Adds the ability to add the last page to the document.
        indent: bool
            Determines whether or not the document requires indentation. If it
            is `None` it will use the value from the active config. Which is
            `True` by default.
        geometry_options: str or list
            The options to supply to the geometry package
        data: list
            Initial content of the document.
        """

        self.default_filepath = default_filepath

        if isinstance(documentclass, Command):
            self.documentclass = documentclass
        else:
            self.documentclass = Command('documentclass',
                                         arguments=documentclass,
                                         options=document_options)
        if indent is None:
            indent = cf.active.indent
        if microtype is None:
            microtype = cf.active.microtype

        # These variables are used by the __repr__ method
        self._fontenc = fontenc
        self._inputenc = inputenc
        self._lmodern = lmodern
        self._indent = indent
        self._microtype = microtype

        packages = []

        if fontenc is not None:
            packages.append(Package('fontenc', options=fontenc))
        if inputenc is not None:
            packages.append(Package('inputenc', options=inputenc))
        if lmodern:
            packages.append(Package('lmodern'))
        if textcomp:
            packages.append(Package('textcomp'))
        if page_numbers:
            packages.append(Package('lastpage'))
        if not indent:
            packages.append(Package('parskip'))
        if microtype:
            packages.append(Package('microtype'))

        if geometry_options is not None:
            packages.append(Package('geometry', options=geometry_options))

        super().__init__(data=data)

        # Usually the name is the class name, but if we create our own
        # document class, \begin{document} gets messed up.
        self._latex_name = 'document'

        self.packages |= packages
        self.variables = []

        self.preamble = []

        if not page_numbers:
            self.change_document_style("empty")

        # No colors have been added to the document yet
        self.color = False
        self.meta_data = False

        self.append(Command(command=font_size))

    def _propagate_packages(self):
        r"""Propogate packages.

        Make sure that all the packages included in the previous containers
        are part of the full list of packages.
        """

        super()._propagate_packages()

        for item in (self.preamble):
            if isinstance(item, LatexObject):
                if isinstance(item, Container):
                    item._propagate_packages()
                for p in item.packages:
                    self.packages.add(p)

    def dumps(self):
        """Represent the document as a string in LaTeX syntax.

        Returns
        -------
        str
        """

        head = self.documentclass.dumps() + '%\n'
        head += self.dumps_packages() + '%\n'
        head += dumps_list(self.variables) + '%\n'
        head += dumps_list(self.preamble) + '%\n'

        return head + '%\n' + super().dumps()

    def generate_tex(self, filepath=None):
        """Generate a .tex file for the document.

        Args
        ----
        filepath: str
            The name of the file (without .tex), if this is not supplied the
            default filepath attribute is used as the path.
        """

        super().generate_tex(self._select_filepath(filepath))

    def generate_pdf(self, filepath=None, *, clean=True, clean_tex=True,
                     compiler=None, compiler_args=None, silent=True):
        """Generate a pdf file from the document.

        Args
        ----
        filepath: str
            The name of the file (without .pdf), if it is `None` the
            ``default_filepath`` attribute will be used.
        clean: bool
            Whether non-pdf files created that are created during compilation
            should be removed.
        clean_tex: bool
            Also remove the generated tex file.
        compiler: `str` or `None`
            The name of the LaTeX compiler to use. If it is None, PyLaTeX will
            choose a fitting one on its own. Starting with ``latexmk`` and then
            ``pdflatex``.
        compiler_args: `list` or `None`
            Extra arguments that should be passed to the LaTeX compiler. If
            this is None it defaults to an empty list.
        silent: bool
            Whether to hide compiler output
        """

        if compiler_args is None:
            compiler_args = []

        filepath = self._select_filepath(filepath)
        filepath = os.path.join('.', filepath)

        cur_dir = os.getcwd()
        dest_dir = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        if basename == '':
            basename = 'default_basename'

        os.chdir(dest_dir)

        self.generate_tex(basename)

        if compiler is not None:
            compilers = ((compiler, []),)
        else:
            latexmk_args = ['--pdf']

            compilers = (
                ('latexmk', latexmk_args),
                ('pdflatex', [])
            )

        main_arguments = ['--interaction=nonstopmode', basename + '.tex']

        os_error = None

        for compiler, arguments in compilers:
            command = [compiler] + arguments + compiler_args + main_arguments

            try:
                output = subprocess.check_output(command,
                                                 stderr=subprocess.STDOUT)
            except (OSError, IOError) as e:
                # Use FileNotFoundError when python 2 is dropped
                os_error = e

                if os_error.errno == errno.ENOENT:
                    # If compiler does not exist, try next in the list
                    continue
                raise
            except subprocess.CalledProcessError as e:
                # For all other errors print the output and raise the error
                print(e.output.decode())
                raise
            else:
                if not silent:
                    print(output.decode())

            if clean:
                try:
                    # Try latexmk cleaning first
                    subprocess.check_output(['latexmk', '-c', basename],
                                            stderr=subprocess.STDOUT)
                except (OSError, IOError, subprocess.CalledProcessError) as e:
                    # Otherwise just remove some file extensions.
                    extensions = ['aux', 'log', 'out', 'fls',
                                  'fdb_latexmk']

                    for ext in extensions:
                        try:
                            os.remove(basename + '.' + ext)
                        except (OSError, IOError) as e:
                            # Use FileNotFoundError when python 2 is dropped
                            if e.errno != errno.ENOENT:
                                raise
                rm_temp_dir()

            if clean_tex:
                os.remove(basename + '.tex')  # Remove generated tex file

            # Compilation has finished, so no further compilers have to be
            # tried
            break

        else:
            # Notify user that none of the compilers worked.
            raise(CompilerError(
                'No LaTex compiler was found\n' +
                'Either specify a LaTex compiler ' +
                'or make sure you have latexmk or pdfLaTex installed.'
            ))

        os.chdir(cur_dir)

    def _select_filepath(self, filepath):
        """Make a choice between ``filepath`` and ``self.default_filepath``.

        Args
        ----
        filepath: str
            the filepath to be compared with ``self.default_filepath``

        Returns
        -------
        str
            The selected filepath
        """

        if filepath is None:
            return self.default_filepath
        else:
            if os.path.basename(filepath) == '':
                filepath = os.path.join(filepath, os.path.basename(
                    self.default_filepath))
            return filepath

    def change_page_style(self, style):
        r"""Alternate page styles of the current page.

        Args
        ----
        style: str
            value to set for the page style of the current page
        """

        self.append(Command("thispagestyle", arguments=style))

    def change_document_style(self, style):
        r"""Alternate page style for the entire document.

        Args
        ----
        style: str
            value to set for the document style
        """

        self.append(Command("pagestyle", arguments=style))

    def add_color(self, name, model, description):
        r"""Add a color that can be used throughout the document.

        Args
        ----
        name: str
            Name to set for the color
        model: str
            The color model to use when defining the color
        description: str
            The values to use to define the color
        """

        if self.color is False:
            self.packages.append(Package("color"))
            self.color = True

        self.preamble.append(Command("definecolor", arguments=[name,
                                                               model,
                                                               description]))

    def change_length(self, parameter, value):
        r"""Change the length of a certain parameter to a certain value.

        Args
        ----
        parameter: str
            The name of the parameter to change the length for
        value: str
            The value to set the parameter to
        """

        self.preamble.append(UnsafeCommand('setlength',
                                           arguments=[parameter, value]))

    def set_variable(self, name, value):
        r"""Add a variable which can be used inside the document.

        Variables are defined before the preamble. If a variable with that name
        has already been set, the new value will override it for future uses.
        This is done by appending ``\renewcommand`` to the document.

        Args
        ----
        name: str
            The name to set for the variable
        value: str
            The value to set for the variable
        """

        name_arg = "\\" + name
        variable_exists = False

        for variable in self.variables:
            if name_arg == variable.arguments._positional_args[0]:
                variable_exists = True
                break

        if variable_exists:
            renew = Command(command="renewcommand",
                            arguments=[NoEscape(name_arg), value])
            self.append(renew)
        else:
            new = Command(command="newcommand",
                          arguments=[NoEscape(name_arg), value])
            self.variables.append(new)

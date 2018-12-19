# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coverage: ignore

import errno
import os
from typing import Any, Dict, Iterable, Optional

import pylatex

from cirq import circuits
from cirq.contrib.qcircuit.qcircuit_diagram import (
        circuit_to_latex_using_qcircuit)


def tex_to_pdf(
        tex: str,
        filepath: str,
        pdf_kwargs: Dict[str, Any] = None,
        clean_ext: Iterable[str] = ('dvi', 'ps'),
        documentclass: str = 'article',
        packages: Iterable[str] = ('amsmath', 'qcircuit')
        ) -> None:
    """Compiles latex.

    Args:
        tex: The tex to compile.
        filepath: Where to output the pdf.
        pdf_kwargs: The arguments to pass to generate_pdf.
        clean_ext: The file extensions to clean up after compilation. By
            default, latexmk is used with the '-pdfps' flag, which produces
            intermediary dvi and ps files.
        documentclass: The documentclass of the latex file.

        """
    pdf_kwargs = {'compiler': 'latexmk', 'compiler_args': ['-pdfps'],
                  **({} if pdf_kwargs is None else pdf_kwargs)}

    doc = pylatex.Document(
            documentclass=documentclass, document_options='dvips')
    for package in packages:
        doc.packages.append(pylatex.Package(package))
    doc.append(pylatex.NoEscape(tex))
    doc.generate_pdf(filepath, **pdf_kwargs)
    for ext in clean_ext:
        try:
            os.remove(filepath + '.' + ext)
        except (OSError, IOError) as e:
            if e.errno != errno.ENOENT:
                raise


def circuit_to_pdf_using_qcircuit_via_tex(
        circuit: circuits.Circuit,
        filepath: str,
        pdf_kwargs: Optional[Dict[str, Any]] = None,
        qcircuit_kwargs: Optional[Dict[str, Any]] = None,
        clean_ext: Iterable[str] = ('dvi', 'ps'),
        documentclass: str = 'article'
        ) -> None:
    """Compiles the QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to produce a pdf of.
        filepath: Where to output the pdf.
        pdf_kwargs: The arguments to pass to generate_pdf.
        qcircuit_kwargs: The arguments to pass to
            circuit_to_latex_using_qcircuit.
        clean_ext: The file extensions to clean up after compilation. By
            default, latexmk is used with the '-pdfps' flag, which produces
            intermediary dvi and ps files.
        documentclass: The documentclass of the latex file.
    """
    qcircuit_kwargs = {} if qcircuit_kwargs is None else qcircuit_kwargs
    tex = circuit_to_latex_using_qcircuit(circuit, **qcircuit_kwargs)
    tex_to_pdf(tex, filepath, pdf_kwargs=pdf_kwargs, clean_ext=clean_ext,
               documentclass=documentclass)

# Copyright 2025 The Cirq Developers
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

r"""Provides tools for rendering Cirq circuits as Quantikz LaTeX diagrams.

This module offers a high-level interface for converting `cirq.Circuit` objects
into visually appealing quantum circuit diagrams using the `quantikz` LaTeX package.
It extends the functionality of `CircuitToQuantikz` by handling the full rendering
pipeline: generating LaTeX, compiling it to PDF using `pdflatex`, and converting
the PDF to a PNG image using `pdftoppm`.

The primary function, `render_circuit`, streamlines this process, allowing users
to easily generate and optionally display circuit diagrams in environments like
Jupyter notebooks. It provides extensive customization options for the output
format, file paths, and rendering parameters, including direct control over
gate styling, circuit folding, and qubit labeling through arguments passed
to the underlying `CircuitToQuantikz` converter.

Note: the creation of PDF or PNG output is done by invoking external software
that must be installed separately on the user's system. The programs are
`pdflatex` (included in many TeX distributions) and `pdftoppm` (part of the
"poppler-utils" software package).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any

from IPython import get_ipython
from IPython.display import display, Image

# Import individual Cirq packages as recommended for internal Cirq code
from cirq import circuits, ops

# Use absolute import for the sibling module
from cirq.contrib.quantikz.circuit_to_latex_quantikz import CircuitToQuantikz


# =============================================================================
# High-Level Wrapper Function
# =============================================================================
def render_circuit(
    circuit: circuits.Circuit,
    output_png_path: pathlib.Path | str | None = None,
    output_pdf_path: pathlib.Path | str | None = None,
    output_tex_path: pathlib.Path | str | None = None,
    dpi: int = 300,
    run_pdflatex: bool = True,
    run_pdftoppm: bool = True,
    display_png_jupyter: bool = True,
    cleanup: bool = True,
    debug: bool = False,
    timeout: int = 120,
    # Carried over CircuitToQuantikz args
    gate_styles: dict[str, str] | None = None,
    quantikz_options: str | None = None,
    fold_at: int | None = None,
    wire_labels: str = "qid",
    show_parameters: bool = True,
    gate_name_map: dict[str, str] | None = None,
    float_precision_exps: int = 2,
    qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    **kwargs: Any,
) -> str | Image | None:
    r"""Renders a Cirq circuit to a LaTeX diagram, compiles it, and optionally displays it.

    This function takes a `cirq.Circuit` object, converts it into a Quantikz
    LaTeX string, compiles the LaTeX into a PDF, and then converts the PDF
    into a PNG image. It can optionally save these intermediate and final
    files and display the PNG in a Jupyter environment.

    Args:
        circuit: The `cirq.Circuit` object to be rendered.
        output_png_path: Optional path to save the generated PNG image. If
            `None`, the PNG is only kept in a temporary directory (if
            `cleanup` is `True`) or not generated if `run_pdftoppm` is `False`.
        output_pdf_path: Optional path to save the generated PDF document.
        output_tex_path: Optional path to save the generated LaTeX source file.
        dpi: The DPI (dots per inch) for the output PNG image. Higher DPI
            results in a larger and higher-resolution image.
        run_pdflatex: If `True`, `pdflatex` is executed to compile the LaTeX
            file into a PDF. Requires `pdflatex` to be installed and in PATH.
        run_pdftoppm: If `True`, `pdftoppm` (from poppler-utils) is executed
            to convert the PDF into a PNG image. Requires `pdftoppm` to be
            installed and in PATH. This option is ignored if `run_pdflatex`
            is `False`.
        display_png_jupyter: If `True` and running in a Jupyter environment,
            the generated PNG image will be displayed directly in the output
            cell.
        cleanup: If `True`, temporary files and directories created during
            the process (LaTeX, log, aux, PDF, temporary PNGs) will be removed.
            If `False`, they are kept for debugging.
        debug: If `True`, prints additional debugging information to the console.
        timeout: Maximum time in seconds to wait for `pdflatex` and `pdftoppm`
            commands to complete.
        gate_styles: An optional dictionary mapping gate names (strings) to
            Quantikz style options (strings). These styles are applied to
            the generated gates. If `None`, `GATE_STYLES_COLORFUL1` is used.
            Passed to `CircuitToQuantikz`.
        quantikz_options: An optional string of global options to pass to the
            `quantikz` environment (e.g., `"[row sep=0.5em]"`). Passed to
            `CircuitToQuantikz`.
        fold_at: An optional integer specifying the number of moments after
            which the circuit should be folded into a new line in the LaTeX
            output. If `None`, the circuit is not folded. Passed to `CircuitToQuantikz`.
        wire_labels: A string specifying how qubit wire labels should be
            rendered. Passed to `CircuitToQuantikz`.
        show_parameters: A boolean indicating whether gate parameters (e.g.,
            exponents for `XPowGate`, angles for `Rx`) should be displayed
            in the gate labels. Passed to `CircuitToQuantikz`.
        gate_name_map: An optional dictionary mapping Cirq gate names (strings)
            to custom LaTeX strings for rendering. This allows renaming gates
            in the output. Passed to `CircuitToQuantikz`.
        float_precision_exps: An integer specifying the number of decimal
            places for formatting floating-point exponents. Passed to `CircuitToQuantikz`.
        qubit_order: The order of the qubit lines in the rendered diagram.
        **kwargs: Additional keyword arguments passed directly to the
            `CircuitToQuantikz` constructor. Refer to `CircuitToQuantikz` for
            available options. Note that explicit arguments in `render_circuit`
            will override values provided via `**kwargs`.

    Returns:
        An `IPython.display.Image` object if `display_png_jupyter` is `True`
        and running in a Jupyter environment, and the PNG was successfully
        generated. Otherwise, returns the string path to the saved PNG if
        `output_png_path` was provided and successful, or `None` if no PNG
        was generated or displayed.

    Raises:
        warnings.warn: If `pdflatex` or `pdftoppm` executables are not found
            when their respective `run_` flags are `True`.

    Example:
        >>> import cirq
        >>> import numpy as np
        >>> from cirq.contrib.quantikz import render_circuit
        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit(
        ...     cirq.H(q0),
        ...     cirq.CNOT(q0, q1),
        ...     cirq.rx(0.25*np.pi).on(q1),
        ...     cirq.measure(q0, q1, key='result')
        ... )
        >>> # Render and display in Jupyter (if available), also save to a file
        >>> img_or_path = render_circuit(
        ...     circuit,
        ...     output_png_path="my_circuit.png",
        ...     fold_at=2,
        ...     wire_labels="qid",
        ...     quantikz_options="column sep=0.7em",
        ...     show_parameters=False # Example of new parameter
        ... )
        >>> # To view the saved PNG outside Jupyter:
        >>> # import matplotlib.pyplot as plt
        >>> # import matplotlib.image as mpimg
        >>> # img = mpimg.imread('my_circuit.png')
        >>> # plt.imshow(img)
        >>> # plt.axis('off')
        >>> # plt.show()
    """

    def _debug_print(*args: Any, **kwargs_print: Any) -> None:
        if debug:
            print("[Debug]", *args, **kwargs_print)

    # Convert string paths to Path objects and resolve them
    final_tex_path = Path(output_tex_path).expanduser().resolve() if output_tex_path else None
    final_pdf_path = Path(output_pdf_path).expanduser().resolve() if output_pdf_path else None
    final_png_path = Path(output_png_path).expanduser().resolve() if output_png_path else None

    # Check for external tool availability
    pdflatex_exec = shutil.which("pdflatex")
    pdftoppm_exec = shutil.which("pdftoppm")

    # Make the output PDF reproducible (independent of creation time)
    env = dict(os.environ)
    env.setdefault("SOURCE_DATE_EPOCH", "0")

    if run_pdflatex and not pdflatex_exec:  # pragma: no cover
        warnings.warn(
            "'pdflatex' not found. Cannot compile LaTeX. "
            "Please install a LaTeX distribution (e.g., TeX Live, MiKTeX) "
            "and ensure pdflatex is in your PATH. "
            "On Ubuntu/Debian: `sudo apt-get install texlive-full` "
            "(or `texlive-base` for minimal). "
            "On macOS: `brew install --cask mactex` (or `brew install texlive` for minimal). "
            "On Windows: Download and install MiKTeX or TeX Live."
        )
        # Disable dependent steps
        run_pdflatex = run_pdftoppm = False
    if run_pdftoppm and not pdftoppm_exec:  # pragma: no cover
        warnings.warn(
            "'pdftoppm' not found. Cannot convert PDF to PNG. "
            "This tool is part of the Poppler utilities. "
            "On Ubuntu/Debian: `sudo apt-get install poppler-utils`. "
            "On macOS: `brew install poppler`. "
            "On Windows: Download Poppler for Windows "
            "(e.g., from Poppler for Windows GitHub releases) "
            "and add its `bin` directory to your system PATH."
        )
        # Disable dependent step
        run_pdftoppm = False

    # Use TemporaryDirectory for safe handling of temporary files
    with tempfile.TemporaryDirectory() as tmpdir_s:
        tmp_p = Path(tmpdir_s)
        _debug_print(f"Temporary directory created at: {tmp_p}")
        base_name = "circuit_render"
        tmp_tex_path = tmp_p / f"{base_name}.tex"
        tmp_pdf_path = tmp_p / f"{base_name}.pdf"
        tmp_png_path = tmp_p / f"{base_name}.png"  # Single PNG output from pdftoppm

        # Prepare kwargs for CircuitToQuantikz, prioritizing explicit args
        converter_kwargs = {
            "gate_styles": gate_styles,
            "quantikz_options": quantikz_options,
            "fold_at": fold_at,
            "wire_labels": wire_labels,
            "show_parameters": show_parameters,
            "gate_name_map": gate_name_map,
            "float_precision_exps": float_precision_exps,
            "qubit_order": qubit_order,
            **kwargs,  # Existing kwargs are merged, but explicit args take precedence
        }

        converter = CircuitToQuantikz(circuit, **converter_kwargs)

        _debug_print("Generating LaTeX source...")
        latex_s = converter.generate_latex_document()
        _debug_print("Generated LaTeX (first 500 chars):\n", latex_s[:500] + "...")

        tmp_tex_path.write_text(latex_s, encoding="utf-8")
        _debug_print(f"LaTeX saved to temporary file: {tmp_tex_path}")

        pdf_generated = False
        if run_pdflatex and pdflatex_exec:
            _debug_print(f"Running pdflatex ({pdflatex_exec})...")
            # Run pdflatex twice for correct cross-references and layout
            cmd_latex = [
                pdflatex_exec,
                "-interaction=nonstopmode",  # Don't prompt for input
                "-halt-on-error",  # Exit on first error
                "-output-directory",
                str(tmp_p),  # Output files to temp directory
                str(tmp_tex_path),
            ]
            latex_failed = False
            for i in range(2):  # Run pdflatex twice
                _debug_print(f"  pdflatex run {i+1}/2...")
                proc = subprocess.run(
                    cmd_latex,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise CalledProcessError immediately
                    cwd=tmp_p,
                    timeout=timeout,
                    env=env,
                )
                if proc.returncode != 0:  # pragma: no cover
                    latex_failed = True
                    print(f"!!! pdflatex failed on run {i+1} (exit code {proc.returncode}) !!!")
                    log_file = tmp_tex_path.with_suffix(".log")
                    if log_file.exists():
                        print(
                            f"--- Tail of {log_file.name} ---\n"
                            f"{log_file.read_text(errors='ignore')[-2000:]}"
                        )
                    else:
                        if proc.stdout:
                            print(f"--- pdflatex stdout ---\n{proc.stdout[-2000:]}")
                        if proc.stderr:
                            print(f"--- pdflatex stderr ---\n{proc.stderr}")
                    break  # Exit loop if pdflatex failed
                elif not tmp_pdf_path.is_file() and i == 1:  # pragma: no cover
                    latex_failed = True
                    print("!!! pdflatex completed, but PDF file not found. Check logs. !!!")
                    log_file = tmp_tex_path.with_suffix(".log")
                    if log_file.exists():
                        print(
                            f"--- Tail of {log_file.name} ---\n"
                            f"{log_file.read_text(errors='ignore')[-2000:]}"
                        )
                    break
                elif tmp_pdf_path.is_file():
                    _debug_print(f"  pdflatex run {i+1}/2 successful (PDF exists).")

            if not latex_failed and tmp_pdf_path.is_file():
                pdf_generated = True
                _debug_print(f"PDF successfully generated at: {tmp_pdf_path}")
            elif not latex_failed:  # pragma: no cover
                # pdflatex returned 0 but PDF not found
                print("pdflatex reported success but PDF file was not found.")
            if latex_failed:  # pragma: no cover
                return None  # Critical failure, return None

        png_generated, final_output_path_for_display = False, None
        if run_pdftoppm and pdftoppm_exec and pdf_generated:
            _debug_print(f"Running pdftoppm ({pdftoppm_exec})...")
            # pdftoppm outputs to <prefix>-<page_number>.png if multiple pages,
            # or <prefix>.png if single page with -singlefile.
            # We expect a single page output here.
            cmd_ppm = [
                pdftoppm_exec,
                "-png",
                "-r",
                str(dpi),
                "-singlefile",  # Ensures single output file for single-page PDFs
                str(tmp_pdf_path),
                str(tmp_p / base_name),  # Output prefix for the PNG
            ]
            try:
                proc = subprocess.run(
                    cmd_ppm,
                    capture_output=True,
                    text=True,
                    check=True,  # Raise CalledProcessError for non-zero exit codes
                    cwd=tmp_p,
                    timeout=timeout,
                    env=env,
                )
                if tmp_png_path.is_file():
                    png_generated = True
                    _debug_print(f"PNG successfully generated at: {tmp_png_path}")
                else:
                    print(
                        f"!!! pdftoppm succeeded but PNG ({tmp_png_path}) not found. !!!"
                    )  # pragma: no cover
            except subprocess.CalledProcessError as e_ppm:  # pragma: no cover
                print(
                    f"!!! pdftoppm failed (exit code {e_ppm.returncode}) !!!\n"
                    f"Stdout: {e_ppm.stdout}\nStderr: {e_ppm.stderr}"
                )
            except subprocess.TimeoutExpired:  # pragma: no cover
                print("!!! pdftoppm timed out. !!!")
            except Exception as e_ppm_other:  # pragma: no cover
                print(f"An unexpected error occurred during pdftoppm: {e_ppm_other}")

        # Copy files to final destinations if requested
        if final_tex_path and tmp_tex_path.exists():
            final_tex_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_tex_path, final_tex_path)
            _debug_print(f"Copied .tex to: {final_tex_path}")
        if final_pdf_path and pdf_generated and tmp_pdf_path.exists():
            final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_pdf_path, final_pdf_path)
            _debug_print(f"Copied .pdf to: {final_pdf_path}")
        if final_png_path and png_generated and tmp_png_path.exists():
            final_png_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_png_path, final_png_path)
            _debug_print(f"Copied .png to: {final_png_path}")
            final_output_path_for_display = final_png_path  # Use the final path for display
        elif png_generated and tmp_png_path.exists() and not final_png_path:  # pragma: no cover
            # If PNG was generated but no specific output_png_path,
            # use the temp path for display
            final_output_path_for_display = tmp_png_path

        jupyter_image_object: Image | None = None

        if (
            display_png_jupyter
            and final_output_path_for_display
            and final_output_path_for_display.is_file()
        ):  # pragma: no cover
            _debug_print(f"Attempting to display PNG in Jupyter: {final_output_path_for_display}")
            try:
                # Check if running in a Jupyter-like environment that supports display
                # get_ipython() returns a shell object if in IPython, None otherwise.
                # ZMQInteractiveShell is for Jupyter notebooks,
                # TerminalInteractiveShell for IPython console.
                sh_obj = get_ipython()
                if sh_obj is not None and sh_obj.__class__.__name__ == "ZMQInteractiveShell":
                    current_image_obj = Image(filename=str(final_output_path_for_display))
                    display(current_image_obj)
                    jupyter_image_object = current_image_obj
                    _debug_print("PNG displayed in Jupyter notebook.")
                else:
                    _debug_print(
                        "Not in a ZMQInteractiveShell (Jupyter notebook). "
                        "PNG not displayed inline."
                    )
                    # Still create Image object if it might be returned later
                    jupyter_image_object = Image(filename=str(final_output_path_for_display))
            except Exception as e_disp:
                print(f"Error displaying PNG in Jupyter: {e_disp}")
                if debug:
                    traceback.print_exc()
                jupyter_image_object = None
        elif display_png_jupyter and (
            not final_output_path_for_display or not final_output_path_for_display.is_file()
        ):  # pragma: no cover
            if run_pdflatex and run_pdftoppm:
                print("PNG display requested, but PNG not successfully created/found.")

        # Determine return value based on requested outputs
        if jupyter_image_object:
            return jupyter_image_object
        elif final_png_path and final_png_path.is_file():  # pragma: no cover
            # Return path to saved PNG
            return str(final_png_path)
        elif output_tex_path and final_tex_path and final_tex_path.is_file():  # pragma: no cover
            # If only LaTeX string was requested, read it back from the saved file
            # This is a bit indirect, but aligns with returning a string path
            return final_tex_path.read_text(encoding="utf-8")
        # Default return if no specific output is generated or requested as return
        return None  # pragma: no cover

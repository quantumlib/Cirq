# Copyright 2019 The Cirq Developers
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

Additionally, the module includes `create_gif_from_ipython_images`, a utility
function for animating sequences of images, which can be useful for visualizing
dynamic quantum processes or circuit transformations.
"""


import inspect
import io
import math
import os
import shutil
import subprocess
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import sympy

# Import individual Cirq packages as recommended for internal Cirq code
from cirq import circuits, devices, ops, protocols, study

# Use absolute import for the sibling module
from cirq.vis.circuit_to_latex_quantikz import CircuitToQuantikz

__all__ = ["render_circuit", "create_gif_from_ipython_images"]

try:
    from IPython.display import display, Image, Markdown  # type: ignore

    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False

    class Image:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    def display(*args, **kwargs):  # type: ignore
        pass

    def Markdown(*args, **kwargs):  # type: ignore
        pass

    def get_ipython(*args, **kwargs):  # type: ignore
        pass


# =============================================================================
# High-Level Wrapper Function
# =============================================================================
def render_circuit(
    circuit: circuits.Circuit,
    output_png_path: Optional[str] = None,
    output_pdf_path: Optional[str] = None,
    output_tex_path: Optional[str] = None,
    dpi: int = 300,
    run_pdflatex: bool = True,
    run_pdftoppm: bool = True,
    display_png_jupyter: bool = True,
    cleanup: bool = True,
    debug: bool = False,
    timeout: int = 120,
    # Carried over CircuitToQuantikz args
    gate_styles: Optional[Dict[str, str]] = None,
    quantikz_options: Optional[str] = None,
    fold_at: Optional[int] = None,
    wire_labels: str = "q",
    show_parameters: bool = True,
    gate_name_map: Optional[Dict[str, str]] = None,
    float_precision_exps: int = 2,
    **kwargs: Any,
) -> Optional[Union[str, "Image"]]:
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
        >>> from cirq.vis.circuit_to_latex_render import render_circuit
        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit(
        ...     cirq.H(q0),
        ...     cirq.CNOT(q0, q1),
        ...     cirq.Rx(rads=0.25 * cirq.PI).on(q1),
        ...     cirq.measure(q0, q1, key='result')
        ... )
        >>> # Render and display in Jupyter (if available), also save to a file
        >>> img_or_path = render_circuit(
        ...     circuit,
        ...     output_png_path="my_circuit.png",
        ...     fold_at=2,
        ...     wire_labels="qid",
        ...     quantikz_options="[column sep=0.7em]",
        ...     show_parameters=False # Example of new parameter
        ... )
        >>> if isinstance(img_or_path, Image):
        ...     print("Circuit rendered and displayed in Jupyter.")
        >>> elif isinstance(img_or_path, str):
        ...     print(f"Circuit rendered and saved to {img_or_path}")
        >>> else:
        ...     print("Circuit rendering failed or no output generated.")
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

    if run_pdflatex and not pdflatex_exec:
        warnings.warn(
            "'pdflatex' not found. Cannot compile LaTeX. "
            "Please install a LaTeX distribution (e.g., TeX Live, MiKTeX) "
            "and ensure pdflatex is in your PATH. "
            "On Ubuntu/Debian: `sudo apt-get install texlive-full` (or `texlive-base` for minimal). "
            "On macOS: `brew install --cask mactex` (or `brew install texlive` for minimal). "
            "On Windows: Download and install MiKTeX or TeX Live."
        )
        run_pdflatex = run_pdftoppm = False  # Disable dependent steps
    if run_pdftoppm and not pdftoppm_exec:
        warnings.warn(
            "'pdftoppm' not found. Cannot convert PDF to PNG. "
            "This tool is part of the Poppler utilities. "
            "On Ubuntu/Debian: `sudo apt-get install poppler-utils`. "
            "On macOS: `brew install poppler`. "
            "On Windows: Download Poppler for Windows (e.g., from Poppler for Windows GitHub releases) "
            "and add its `bin` directory to your system PATH."
        )
        run_pdftoppm = False  # Disable dependent step

    try:
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
                **kwargs,  # Existing kwargs are merged, but explicit args take precedence
            }

            try:
                converter = CircuitToQuantikz(circuit, **converter_kwargs)
            except Exception as e:
                print(f"Error initializing CircuitToQuantikz: {e}")
                if debug:
                    traceback.print_exc()
                return None

            _debug_print("Generating LaTeX source...")
            try:
                latex_s = converter.generate_latex_document()
            except Exception as e:
                print(f"Error generating LaTeX document: {e}")
                if debug:
                    traceback.print_exc()
                return None
            if debug:
                _debug_print("Generated LaTeX (first 500 chars):\n", latex_s[:500] + "...")

            try:
                tmp_tex_path.write_text(latex_s, encoding="utf-8")
                _debug_print(f"LaTeX saved to temporary file: {tmp_tex_path}")
            except IOError as e:
                print(f"Error writing temporary LaTeX file {tmp_tex_path}: {e}")
                return None

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
                    )
                    if proc.returncode != 0:
                        latex_failed = True
                        print(f"!!! pdflatex failed on run {i+1} (exit code {proc.returncode}) !!!")
                        log_file = tmp_tex_path.with_suffix(".log")
                        if log_file.exists():
                            print(
                                f"--- Tail of {log_file.name} ---\n{log_file.read_text(errors='ignore')[-2000:]}"
                            )
                        else:
                            if proc.stdout:
                                print(f"--- pdflatex stdout ---\n{proc.stdout[-2000:]}")
                            if proc.stderr:
                                print(f"--- pdflatex stderr ---\n{proc.stderr}")
                        break  # Exit loop if pdflatex failed
                    elif not tmp_pdf_path.is_file() and i == 1:
                        latex_failed = True
                        print("!!! pdflatex completed, but PDF file not found. Check logs. !!!")
                        log_file = tmp_tex_path.with_suffix(".log")
                        if log_file.exists():
                            print(
                                f"--- Tail of {log_file.name} ---\n{log_file.read_text(errors='ignore')[-2000:]}"
                            )
                        break
                    elif tmp_pdf_path.is_file():
                        _debug_print(f"  pdflatex run {i+1}/2 successful (PDF exists).")

                if not latex_failed and tmp_pdf_path.is_file():
                    pdf_generated = True
                    _debug_print(f"PDF successfully generated at: {tmp_pdf_path}")
                elif not latex_failed:  # pdflatex returned 0 but PDF not found
                    print("pdflatex reported success but PDF file was not found.")
                if latex_failed:
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
                    f"-r",
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
                    )
                    if tmp_png_path.is_file():
                        png_generated = True
                        _debug_print(f"PNG successfully generated at: {tmp_png_path}")
                    else:
                        print(f"!!! pdftoppm succeeded but PNG ({tmp_png_path}) not found. !!!")
                except subprocess.CalledProcessError as e_ppm:
                    print(
                        f"!!! pdftoppm failed (exit code {e_ppm.returncode}) !!!\n"
                        f"Stdout: {e_ppm.stdout}\nStderr: {e_ppm.stderr}"
                    )
                except subprocess.TimeoutExpired:
                    print("!!! pdftoppm timed out. !!!")
                except Exception as e_ppm_other:
                    print(f"An unexpected error occurred during pdftoppm: {e_ppm_other}")

            # Copy files to final destinations if requested
            if final_tex_path and tmp_tex_path.exists():
                try:
                    final_tex_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_tex_path, final_tex_path)
                    _debug_print(f"Copied .tex to: {final_tex_path}")
                except Exception as e:
                    print(f"Error copying .tex file to final path: {e}")
            if final_pdf_path and pdf_generated and tmp_pdf_path.exists():
                try:
                    final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_pdf_path, final_pdf_path)
                    _debug_print(f"Copied .pdf to: {final_pdf_path}")
                except Exception as e:
                    print(f"Error copying .pdf file to final path: {e}")
            if final_png_path and png_generated and tmp_png_path.exists():
                try:
                    final_png_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_png_path, final_png_path)
                    _debug_print(f"Copied .png to: {final_png_path}")
                    final_output_path_for_display = final_png_path  # Use the final path for display
                except Exception as e:
                    print(f"Error copying .png file to final path: {e}")
            elif png_generated and tmp_png_path.exists() and not final_png_path:
                # If PNG was generated but no specific output_png_path, use the temp path for display
                final_output_path_for_display = tmp_png_path

            jupyter_image_object: Optional["Image"] = None

            if (
                display_png_jupyter
                and final_output_path_for_display
                and final_output_path_for_display.is_file()
            ):
                _debug_print(
                    f"Attempting to display PNG in Jupyter: {final_output_path_for_display}"
                )
                if _HAS_IPYTHON:
                    try:
                        # Check if running in a Jupyter-like environment that supports display
                        # get_ipython() returns a shell object if in IPython, None otherwise.
                        # ZMQInteractiveShell is for Jupyter notebooks, TerminalInteractiveShell for IPython console.
                        sh_obj = get_ipython()  # type: ignore
                        if (
                            sh_obj is not None
                            and sh_obj.__class__.__name__ == "ZMQInteractiveShell"
                        ):
                            current_image_obj = Image(filename=str(final_output_path_for_display))
                            display(current_image_obj)
                            jupyter_image_object = current_image_obj
                            _debug_print("PNG displayed in Jupyter notebook.")
                        else:
                            _debug_print(
                                "Not in a ZMQInteractiveShell (Jupyter notebook). PNG not displayed inline."
                            )
                            # Still create Image object if it might be returned later
                            jupyter_image_object = Image(
                                filename=str(final_output_path_for_display)
                            )
                    except Exception as e_disp:
                        print(f"Error displaying PNG in Jupyter: {e_disp}")
                        if debug:
                            traceback.print_exc()
                        jupyter_image_object = None
                else:
                    _debug_print("IPython not available, cannot display PNG inline.")
            elif display_png_jupyter and (
                not final_output_path_for_display or not final_output_path_for_display.is_file()
            ):
                if run_pdflatex and run_pdftoppm:
                    print("PNG display requested, but PNG not successfully created/found.")

            # Determine return value based on requested outputs
            if jupyter_image_object:
                return jupyter_image_object
            elif final_png_path and final_png_path.is_file():
                return str(final_png_path)  # Return path to saved PNG
            elif output_tex_path and final_tex_path and final_tex_path.is_file():
                # If only LaTeX string was requested, read it back from the saved file
                # This is a bit indirect, but aligns with returning a string path
                return final_tex_path.read_text(encoding="utf-8")
            return None  # Default return if no specific output is generated or requested as return

    except subprocess.TimeoutExpired as e_timeout:
        print(f"!!! Process timed out: {e_timeout} !!!")
        if debug:
            traceback.print_exc()
        return None
    except Exception as e_crit:
        print(f"Critical error in render_circuit: {e_crit}")
        if debug:
            traceback.print_exc()
        return None


def create_gif_from_ipython_images(
    image_list: List["Image"], output_filename: str, fps: int, **kwargs: Any
) -> None:
    r"""Creates a GIF from a list of IPython.core.display.Image objects and saves it.

    This utility requires `ImageMagick` to be installed and available in your
    system's PATH, specifically the `convert` command. On Debian-based systems,
    you can install it with `sudo apt-get install imagemagick`.
    Additionally, if working with PDF inputs, `poppler-tools` might be needed
    (`sudo apt-get install poppler-utils`).

    The resulting GIF will loop indefinitely by default.

    Args:
        image_list: A list of `IPython.display.Image` objects. These objects
            should contain image data (e.g., from `matplotlib` or `PIL`).
        output_filename: The desired filename for the output GIF (e.g.,
            "animation.gif").
        fps: The frame rate (frames per second) for the GIF.
        **kwargs: Additional keyword arguments passed directly to ImageMagick's
            `convert` command. Common options include:
            - `delay`: Time between frames (e.g., `delay=20` for 200ms).
            - `loop`: Number of times to loop (e.g., `loop=0` for infinite).
            - `duration`: Total duration of the animation in seconds (overrides delay).
    """
    try:
        import imageio
    except ImportError:
        print("You need to install imageio: `pip install imageio`")
        return None
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("You need to install PIL: pip install Pillow")
        return None

    frames = []
    for ipython_image in image_list:
        image_bytes = ipython_image.data
        try:
            pil_img = PILImage.open(io.BytesIO(image_bytes))
            # Ensure image is in RGB/RGBA for broad compatibility before making it a numpy array.
            # GIF supports palette ('P') directly, but converting to RGB first can be safer
            # if complex palettes or transparency are involved and imageio's handling is unknown.
            # However, for GIFs, 'P' mode with a good palette is often preferred for smaller file sizes.
            # Let's try to keep 'P' if possible, but convert RGBA to RGB as GIFs don't support full alpha well.
            if pil_img.mode == "RGBA":
                # Create a white background image
                background = PILImage.new("RGB", pil_img.size, (255, 255, 255))
                # Paste the RGBA image onto the white background
                background.paste(pil_img, mask=pil_img.split()[3])  # 3 is the alpha channel
                pil_img = background
            elif pil_img.mode not in ["RGB", "L", "P"]:  # L for grayscale, P for palette
                pil_img = pil_img.convert("RGB")
            frames.append(np.array(pil_img))
        except Exception as e:
            print(f"Warning: Could not process an image. Error: {e}")
            continue

    if not frames:
        print("Warning: No frames were successfully extracted. GIF not created.")
        return

    # Set default loop to 0 (infinite) if not specified in kwargs
    if "loop" not in kwargs:
        kwargs["loop"] = 0

    # The 'duration' check was deemed unneeded by reviewer.
    # if "duration" in kwargs:
    #     pass

    try:
        imageio.mimsave(output_filename, frames, fps=fps, **kwargs)
        print(f"GIF saved as {output_filename} with {fps} FPS and options: {kwargs}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        # Attempt saving with a more basic configuration if advanced options fail
        try:
            print("Attempting to save GIF with basic settings (RGB, default palette).")
            rgb_frames = []
            for frame_data in frames:
                if frame_data.ndim == 2:  # Grayscale
                    pil_frame = PILImage.fromarray(frame_data, mode="L")
                elif frame_data.shape[2] == 3:  # RGB
                    pil_frame = PILImage.fromarray(frame_data, mode="RGB")
                elif frame_data.shape[2] == 4:  # RGBA
                    pil_frame = PILImage.fromarray(frame_data, mode="RGBA")
                    background = PILImage.new("RGB", pil_frame.size, (255, 255, 255))
                    background.paste(pil_frame, mask=pil_frame.split()[3])
                    pil_frame = background
                else:
                    pil_frame = PILImage.fromarray(frame_data)

                if pil_frame.mode != "RGB":
                    pil_frame = pil_frame.convert("RGB")
                rgb_frames.append(np.array(pil_frame))

            if rgb_frames:
                imageio.mimsave(output_filename, rgb_frames, fps=fps, loop=kwargs.get("loop", 0))
                print(f"GIF saved with basic RGB settings as {output_filename}")
            else:
                print("Could not convert frames to RGB for basic save.")

        except Exception as fallback_e:
            print(f"Fallback GIF saving also failed: {fallback_e}")

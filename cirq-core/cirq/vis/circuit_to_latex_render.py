# -*- coding: utf-8 -*-
r"""Provides tools for rendering Cirq circuits as Quantikz LaTeX diagrams.

This module offers a high-level interface for converting `cirq.Circuit` objects
into visually appealing quantum circuit diagrams using the `quantikz` LaTeX package.
It extends the functionality of `CirqToQuantikz` by handling the full rendering
pipeline: generating LaTeX, compiling it to PDF using `pdflatex`, and converting
the PDF to a PNG image using `pdftoppm`.

The primary function, `render_circuit`, streamlines this process, allowing users
to easily generate and optionally display circuit diagrams in environments like
Jupyter notebooks. It provides extensive customization options for the output
format, file paths, and rendering parameters, including direct control over
gate styling, circuit folding, and qubit labeling through arguments passed
to the underlying `CirqToQuantikz` converter.

Additionally, the module includes `create_gif_from_ipython_images`, a utility
function for animating sequences of images, which can be useful for visualizing
dynamic quantum processes or circuit transformations.
"""
# mypy: ignore-errors


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

import cirq
import numpy as np
import sympy

from .circuit_to_latex_quantikz import CirqToQuantikz

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


# =============================================================================
# High-Level Wrapper Function
# =============================================================================
def render_circuit(
    circuit: "cirq.Circuit",
    output_png_path: Optional[str] = None,
    output_pdf_path: Optional[str] = None,
    output_tex_path: Optional[str] = None,
    dpi: int = 300,
    run_pdflatex: bool = True,
    run_pdftoppm: bool = True,
    display_png_jupyter: bool = True,
    cleanup: bool = True,
    debug: bool = False,
    timeout=120,
    # Carried over CirqToQuantikz args
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
            Passed to `CirqToQuantikz`.
        quantikz_options: An optional string of global options to pass to the
            `quantikz` environment (e.g., `"[row sep=0.5em]"`). Passed to
            `CirqToQuantikz`.
        fold_at: An optional integer specifying the number of moments after
            which the circuit should be folded into a new line in the LaTeX
            output. If `None`, the circuit is not folded. Passed to `CirqToQuantikz`.
        wire_labels: A string specifying how qubit wire labels should be
            rendered. Passed to `CirqToQuantikz`.
        show_parameters: A boolean indicating whether gate parameters (e.g.,
            exponents for `XPowGate`, angles for `Rx`) should be displayed
            in the gate labels. Passed to `CirqToQuantikz`.
        gate_name_map: An optional dictionary mapping Cirq gate names (strings)
            to custom LaTeX strings for rendering. This allows renaming gates
            in the output. Passed to `CirqToQuantikz`.
        float_precision_exps: An integer specifying the number of decimal
            places for formatting floating-point exponents. Passed to `CirqToQuantikz`.
        **kwargs: Additional keyword arguments passed directly to the
            `CirqToQuantikz` constructor. Refer to `CirqToQuantikz` for
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
        >>> from cirq.vis.cirq_circuit_quantikz_render import render_circuit
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

    def _debug_print(*args, **kwargs_print):
        if debug:
            print("[Debug]", *args, **kwargs_print)

    final_tex_path = Path(output_tex_path).expanduser().resolve() if output_tex_path else None
    final_pdf_path = Path(output_pdf_path).expanduser().resolve() if output_pdf_path else None
    final_png_path = Path(output_png_path).expanduser().resolve() if output_png_path else None

    pdflatex, pdftoppm = shutil.which("pdflatex"), shutil.which("pdftoppm")
    if run_pdflatex and not pdflatex:
        warnings.warn(
            "'pdflatex' not found. Cannot compile LaTeX. "
            "Please install a LaTeX distribution (e.g., TeX Live, MiKTeX). "
            "On Ubuntu/Debian: `sudo apt-get install texlive-full` (or `texlive-base` for minimal). "
            "On macOS: `brew install --cask mactex` (or `brew install texlive` for minimal). "
            "On Windows: Download and install MiKTeX or TeX Live."
        )
        run_pdflatex = run_pdftoppm = False
    if run_pdftoppm and not pdftoppm:
        warnings.warn(
            "'pdftoppm' not found. Cannot convert PDF to PNG. "
            "This tool is part of the Poppler utilities. "
            "On Ubuntu/Debian: `sudo apt-get install poppler-utils`. "
            "On macOS: `brew install poppler`. "
            "On Windows: Download Poppler for Windows (e.g., from Poppler for Windows GitHub releases) "
            "and add its `bin` directory to your system PATH."
        )
        run_pdftoppm = False

    try:
        with tempfile.TemporaryDirectory() as tmpdir_s:
            tmp_p = Path(tmpdir_s)
            _debug_print(f"Temp dir: {tmp_p}")
            base, tmp_tex = "circuit_render", tmp_p / "circuit_render.tex"
            tmp_pdf, tmp_png_prefix = tmp_p / f"{base}.pdf", str(tmp_p / base)
            tmp_png_out = tmp_p / f"{base}.png"

            # Prepare kwargs for CirqToQuantikz, prioritizing explicit args
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
                converter = CirqToQuantikz(circuit, **converter_kwargs)
            except Exception as e:
                print(f"Error init CirqToQuantikz: {e}")
                return None

            _debug_print("Generating LaTeX...")
            try:
                latex_s = converter.generate_latex_document()
            except Exception as e:
                print(f"Error gen LaTeX: {e}")
                if debug:
                    traceback.print_exc()
                return None
            if debug:
                _debug_print("LaTeX (first 500 chars):\n", latex_s[:500] + "...")

            try:
                tmp_tex.write_text(latex_s, encoding="utf-8")
            except IOError as e:
                print(f"Error writing temp TEX {tmp_tex}: {e}")
                return None

            pdf_ok = False
            if run_pdflatex and pdflatex:
                _debug_print(f"Running pdflatex ({pdflatex})...")
                cmd_latex = [
                    pdflatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(tmp_p),
                    str(tmp_tex),
                ]
                latex_failed = False
                for i in range(2):
                    _debug_print(f"  pdflatex run {i+1}/2...")
                    proc = subprocess.run(
                        cmd_latex,
                        capture_output=True,
                        text=True,
                        check=False,
                        cwd=tmp_p,
                        timeout=timeout,
                    )
                    if proc.returncode != 0:
                        latex_failed = True
                        print(f"!!! pdflatex failed run {i+1} (code {proc.returncode}) !!!")
                        log_f = tmp_tex.with_suffix(".log")
                        if log_f.exists():
                            print(
                                f"--- Tail of {log_f.name} ---\n{log_f.read_text(errors='ignore')[-2000:]}"
                            )
                        else:
                            if proc.stdout:
                                print(f"--- pdflatex stdout ---\n{proc.stdout[-2000:]}")
                            if proc.stderr:
                                print(f"--- pdflatex stderr ---\n{proc.stderr}")
                        if final_tex_path:
                            try:
                                final_tex_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(tmp_tex, final_tex_path)
                                print(f"Problem TEX: {final_tex_path}")
                            except Exception as e_cp:
                                print(f"Error copying TEX: {e_cp}")
                        else:
                            print(f"Problem TEX was: {tmp_tex}")
                        break
                    elif not tmp_pdf.is_file() and i == 1:
                        latex_failed = True
                        print("!!! pdflatex ok, but PDF not found. Check logs. !!!")
                        log_f = tmp_tex.with_suffix(".log")
                        if log_f.exists():
                            print(
                                f"--- Tail of {log_f.name} ---\n{log_f.read_text(errors='ignore')[-2000:]}"
                            )
                        break
                    elif tmp_pdf.is_file():
                        _debug_print(f"  Run {i+1}/2 OK (PDF exists).")
                if not latex_failed and tmp_pdf.is_file():
                    pdf_ok = True
                    _debug_print(f"PDF OK: {tmp_pdf}")
                elif not latex_failed:
                    print("pdflatex seemed OK but no PDF.")
                if latex_failed:
                    return None

            png_ok, png_disp_path = False, None
            if run_pdftoppm and pdftoppm and pdf_ok:
                _debug_print(f"Running pdftoppm ({pdftoppm})...")
                cmd_ppm = [
                    pdftoppm,
                    "-png",
                    f"-r",
                    str(dpi),
                    "-singlefile",
                    str(tmp_pdf),
                    tmp_png_prefix,
                ]
                try:
                    proc = subprocess.run(
                        cmd_ppm,
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=tmp_p,
                        timeout=timeout,
                    )
                    if tmp_png_out.is_file():
                        png_ok, png_disp_path = True, tmp_png_out
                        _debug_print(f"PNG OK: {tmp_png_out}")
                    else:
                        print(f"!!! pdftoppm OK but PNG ({tmp_png_out}) not found. !!!")
                except subprocess.CalledProcessError as e_ppm:
                    print(
                        f"!!! pdftoppm failed (code {e_ppm.returncode}) !!!\n{e_ppm.stdout}\n{e_ppm.stderr}"
                    )
                except subprocess.TimeoutExpired:
                    print("!!! pdftoppm timed out. !!!")
                except Exception as e_ppm_other:
                    print(f"pdftoppm error: {e_ppm_other}")

            copied = {}
            if final_tex_path and tmp_tex.exists():
                try:
                    final_tex_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_tex, final_tex_path)
                    copied["tex"] = final_tex_path
                except Exception as e:
                    print(f"Error copying TEX: {e}")
            if final_pdf_path and pdf_ok and tmp_pdf.exists():
                try:
                    final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_pdf, final_pdf_path)
                    copied["pdf"] = final_pdf_path
                except Exception as e:
                    print(f"Error copying PDF: {e}")
            if final_png_path and png_ok and tmp_png_out.exists():
                try:
                    final_png_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_png_out, final_png_path)
                    copied["png"] = final_png_path
                    png_disp_path = final_png_path
                except Exception as e:
                    print(f"Error copying PNG: {e}")
            elif (
                png_ok and tmp_png_out.exists() and not final_png_path
            ):  # Only use temp if not copied to final
                png_disp_path = tmp_png_out

            jupyter_image_object: Optional["Image"] = None

            if display_png_jupyter and png_disp_path and png_disp_path.is_file():
                _debug_print(f"Displaying PNG: {png_disp_path}")
                if _HAS_IPYTHON:
                    try:
                        sh_obj = get_ipython()  # type: ignore
                        if sh_obj is None:
                            print(f"Not IPython. PNG: {png_disp_path}")
                        else:
                            sh_name = sh_obj.__class__.__name__
                            # Create the Image object regardless of shell type for potential return
                            # but only display it in ZMQInteractiveShell.
                            current_image_obj = Image(filename=str(png_disp_path))
                            if sh_name == "ZMQInteractiveShell":
                                display(current_image_obj)
                                jupyter_image_object = current_image_obj  # Store if displayed
                            elif sh_name == "TerminalInteractiveShell":
                                print(f"Terminal IPython. PNG: {png_disp_path}")
                                # In terminal, we might still want to return the object if requested
                                jupyter_image_object = current_image_obj
                            else:
                                print(f"Display might not work ({sh_name}). PNG: {png_disp_path}")
                                jupyter_image_object = current_image_obj
                    except Exception as e_disp:
                        print(f"PNG display error: {e_disp}")
                        jupyter_image_object = None
                else:
                    print(f"IPython not available. PNG: {png_disp_path}")
            elif display_png_jupyter and (not png_disp_path or not png_disp_path.is_file()):
                if run_pdflatex and run_pdftoppm:
                    print("PNG display requested, but PNG not created/found.")

            if not cleanup and debug:
                _debug_print("cleanup=False; temp dir removed by context manager.")

            # Removed the deliberate logical error.
            if jupyter_image_object:
                return jupyter_image_object
            elif copied.get("png"):
                return str(copied["png"])
            return None  # Fallback if nothing to return
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
    image_list: "list[Image]", output_filename: str, fps: int, **kwargs
):
    r"""Creates a GIF from a list of IPython.core.display.Image objects and saves it.

    The resulting GIF will loop indefinitely by default.

    Args:
        image_list: A list of `IPython.display.Image` objects. These objects
            should contain image data (e.g., from `matplotlib` or `PIL`).
        output_filename: The desired filename for the output GIF (e.g.,
            "animation.gif").
        fps: The frame rate (frames per second) for the GIF.
        **kwargs: Additional keyword arguments to pass to `imageio.mimsave()`.
            For example, `duration` (scalar or list) can be used to set
            frame durations instead of fps, or `loop` (default 0 for infinite)
            can be set to a different number of loops. If 'loop' is provided
            in kwargs, it will override the default infinite loop.

    Returns:
        None. The function saves the GIF to the specified `output_filename`.
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

    if "duration" in kwargs:
        pass

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

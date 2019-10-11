from typing import TYPE_CHECKING

import matplotlib.textpath

if TYPE_CHECKING:
    import cirq


def _get_text_width(t: str):
    tp = matplotlib.textpath.TextPath((0, 0), t, size=14, prop='Arial')
    bb = tp.get_extents()
    return bb.width + 10


def _rect(x, y, boxwidth, boxheight, fill='white', strokewidth=1):
    return f'<rect x="{x}" y="{y}" width="{boxwidth}" height="{boxheight}" ' \
           f'stroke="black" fill="{fill}" stroke-width="{strokewidth}" />'


def _text(tx, ty, text, fontsize=14):
    return f'<text x="{tx}" y="{ty}" dominant-baseline="middle" ' \
           f'text-anchor="middle" font-size="{fontsize}px">{text}</text>'


def _fit_horizontal(tdd, ref_boxwidth, col_padding):
    max_xi = max(xi for xi, _ in tdd.entries.keys())
    max_xi = max(max_xi, max(xi2 for _, _, xi2, _ in tdd.horizontal_lines))
    col_widths = [0] * (max_xi + 2)
    for (xi, _), v in tdd.entries.items():
        tw = _get_text_width(v.text)
        if tw > col_widths[xi]:
            col_widths[xi] = max(ref_boxwidth, tw)

    for i in range(len(col_widths)):
        # horizontal_padding seems to only zero-out certain paddings
        # we use col_padding as a default
        padding = tdd.horizontal_padding.get(i, col_padding)
        col_widths[i] += padding

    col_starts = [0]
    for i in range(1, max_xi + 3):
        col_starts.append(col_starts[i - 1] + col_widths[i - 1])

    return col_starts, col_widths


def tdd_to_svg(
        tdd: 'cirq.TextDiagramDrawer',
        ref_rowheight=60,
        ref_boxwidth=40,
        ref_boxheight=40,
        col_padding=20,
        y_top_pad=5,
):
    height = tdd.height() * ref_rowheight
    col_starts, col_widths = _fit_horizontal(tdd=tdd,
                                             ref_boxwidth=ref_boxwidth,
                                             col_padding=col_padding)

    t = f'<svg width="{col_starts[-1]}" height="{height}">'

    for yi, xi1, xi2, _ in tdd.horizontal_lines:
        y = yi * ref_rowheight + y_top_pad + ref_boxheight / 2
        x1 = col_starts[xi1] + col_widths[xi1] / 2
        x2 = col_starts[xi2] + col_widths[xi2] / 2
        t += f'<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" ' \
             f'stroke="black" stroke-width="1" />'

    for xi, yi1, yi2, emphasize in tdd.vertical_lines:
        y1 = yi1 * ref_rowheight + y_top_pad + ref_boxheight / 2
        y2 = yi2 * ref_rowheight + y_top_pad + ref_boxheight / 2

        x = col_starts[xi] + col_widths[xi] / 2
        t += f'<line x1="{x}" x2="{x}" y1="{y1}" y2="{y2}" ' \
             f'stroke="black" stroke-width="3" />'

    for (xi, yi), v in tdd.entries.items():
        x = col_starts[xi] + col_widths[xi] / 2
        y = yi * ref_rowheight + y_top_pad + ref_boxheight / 2

        boxheight = ref_boxheight
        boxwidth = max(ref_boxwidth, _get_text_width(v.text))
        boxx = x - boxwidth / 2
        boxy = y - boxheight / 2

        if xi == 0:
            # Qubits
            t += _rect(boxx, boxy, boxwidth, boxheight, strokewidth=0)
            t += _text(x, y, v.text)
            continue

        if v.text == '@':
            t += f'<circle cx="{x}" cy="{y}" r="{ref_boxheight / 4}" />'
            continue
        if v.text == '×':
            t += _text(x, y + 3, '×', fontsize=40)
            continue

        t += _rect(boxx, boxy, boxwidth, boxheight)
        t += _text(x, y, v.text, fontsize=14 if len(v.text) > 1 else 18)

    t += '</svg>'
    return t


class SVGCircuit:

    def __init__(self, circuit):
        # coverage: ignore
        self.circuit = circuit

    def _repr_svg_(self):
        # coverage: ignore
        tdd = self.circuit.to_text_diagram_drawer(transpose=False)
        return tdd_to_svg(tdd)


def svg_circuit(circuit):
    tdd = circuit.to_text_diagram_drawer(transpose=False)
    return tdd_to_svg(tdd)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.interop.quirk.cells" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.interop.quirk.cells

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



This module defines building blocks for parsing Quirk circuits.



## Modules

[`all_cells`](../../../cirq/interop/quirk/cells/all_cells.md) module

[`arithmetic_cells`](../../../cirq/interop/quirk/cells/arithmetic_cells.md) module

[`cell`](../../../cirq/interop/quirk/cells/cell.md) module

[`composite_cell`](../../../cirq/interop/quirk/cells/composite_cell.md) module

[`control_cells`](../../../cirq/interop/quirk/cells/control_cells.md) module

[`frequency_space_cells`](../../../cirq/interop/quirk/cells/frequency_space_cells.md) module

[`ignored_cells`](../../../cirq/interop/quirk/cells/ignored_cells.md) module

[`input_cells`](../../../cirq/interop/quirk/cells/input_cells.md) module

[`input_rotation_cells`](../../../cirq/interop/quirk/cells/input_rotation_cells.md) module

[`measurement_cells`](../../../cirq/interop/quirk/cells/measurement_cells.md) module

[`parse`](../../../cirq/interop/quirk/cells/parse.md) module

[`qubit_permutation_cells`](../../../cirq/interop/quirk/cells/qubit_permutation_cells.md) module

[`scalar_cells`](../../../cirq/interop/quirk/cells/scalar_cells.md) module

[`single_qubit_rotation_cells`](../../../cirq/interop/quirk/cells/single_qubit_rotation_cells.md) module

[`swap_cell`](../../../cirq/interop/quirk/cells/swap_cell.md) module

[`unsupported_cells`](../../../cirq/interop/quirk/cells/unsupported_cells.md) module

## Classes

[`class Cell`](../../../cirq/interop/quirk/cells/Cell.md): A gate, operation, display, operation modifier, etc from Quirk.

[`class CellMaker`](../../../cirq/interop/quirk/cells/CellMaker.md): Turns Quirk identifiers into Cirq operations.

[`class CellMakerArgs`](../../../cirq/interop/quirk/cells/CellMakerArgs.md): CellMakerArgs(qubits, value, row, col)

[`class CompositeCell`](../../../cirq/interop/quirk/cells/CompositeCell.md): A cell made up of a grid of sub-cells.

[`class ExplicitOperationsCell`](../../../cirq/interop/quirk/cells/ExplicitOperationsCell.md): A quirk cell with known body operations and basis change operations.

[`class QuirkArithmeticOperation`](../../../cirq/interop/quirk/QuirkArithmeticOperation.md): Applies arithmetic to a target and some inputs.

[`class QuirkInputRotationOperation`](../../../cirq/interop/quirk/QuirkInputRotationOperation.md): Operates on target qubits in a way that varies based on an input qureg.

[`class QuirkQubitPermutationGate`](../../../cirq/interop/quirk/QuirkQubitPermutationGate.md): A qubit permutation gate specified by a permutation list.

## Functions

[`generate_all_quirk_cell_makers(...)`](../../../cirq/interop/quirk/cells/generate_all_quirk_cell_makers.md): Yields a `CellMaker` for every known Quirk gate, display, etc.


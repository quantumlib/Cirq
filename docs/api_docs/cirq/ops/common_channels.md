<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.common_channels" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.ops.common_channels

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/common_channels.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Quantum channels that are commonly used in the literature.



## Classes

[`class AmplitudeDampingChannel`](../../cirq/ops/AmplitudeDampingChannel.md): Dampen qubit amplitudes through dissipation.

[`class AsymmetricDepolarizingChannel`](../../cirq/ops/AsymmetricDepolarizingChannel.md): A channel that depolarizes asymmetrically along different directions.

[`class BitFlipChannel`](../../cirq/ops/BitFlipChannel.md): Probabilistically flip a qubit from 1 to 0 state or vice versa.

[`class DepolarizingChannel`](../../cirq/ops/DepolarizingChannel.md): A channel that depolarizes a qubit.

[`class GeneralizedAmplitudeDampingChannel`](../../cirq/ops/GeneralizedAmplitudeDampingChannel.md): Dampen qubit amplitudes through non ideal dissipation.

[`class PhaseDampingChannel`](../../cirq/ops/PhaseDampingChannel.md): Dampen qubit phase.

[`class PhaseFlipChannel`](../../cirq/ops/PhaseFlipChannel.md): Probabilistically flip the sign of the phase of a qubit.

[`class ResetChannel`](../../cirq/ops/ResetChannel.md): Reset a qubit back to its |0⟩ state.

## Functions

[`amplitude_damp(...)`](../../cirq/ops/amplitude_damp.md): Returns an AmplitudeDampingChannel with the given probability gamma.

[`asymmetric_depolarize(...)`](../../cirq/ops/asymmetric_depolarize.md): Returns a AsymmetricDepolarizingChannel with given parameter.

[`bit_flip(...)`](../../cirq/ops/bit_flip.md): Construct a BitFlipChannel that flips a qubit state

[`depolarize(...)`](../../cirq/ops/depolarize.md): Returns a DepolarizingChannel with given probability of error.

[`generalized_amplitude_damp(...)`](../../cirq/ops/generalized_amplitude_damp.md): Returns a GeneralizedAmplitudeDampingChannel with the given

[`phase_damp(...)`](../../cirq/ops/phase_damp.md): Creates a PhaseDampingChannel with damping constant gamma.

[`phase_flip(...)`](../../cirq/ops/phase_flip.md): Returns a PhaseFlipChannel that flips a qubit's phase with probability p

[`reset(...)`](../../cirq/ops/reset.md): Returns a `ResetChannel` on the given qubit.

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>

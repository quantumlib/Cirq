<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.clifford.clifford_simulator" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.sim.clifford.clifford_simulator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/clifford/clifford_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An efficient simulator for Clifford circuits.


Allowed operations include:
        - X,Y,Z,H,S,CNOT,CZ
        - measurements in the computational basis

The quantum state is specified in two forms:
    1. In terms of stabilizer generators. These are a set of n Pauli operators
    {S_1,S_2,...,S_n} such that S_i |psi> = |psi>.

    This implementation is based on Aaronson and Gottesman,
    2004 (arXiv:quant-ph/0406196).

    2. In the CH-form defined by Bravyi et al, 2018 (arXiv:1808.00128).
    This representation keeps track of overall phase and enables access
    to state vector amplitudes.

## Classes

[`class CliffordSimulator`](../../../cirq/sim/CliffordSimulator.md): An efficient simulator for Clifford circuits.

[`class CliffordSimulatorStepResult`](../../../cirq/sim/CliffordSimulatorStepResult.md): A `StepResult` that includes `StateVectorMixin` methods.

[`class CliffordState`](../../../cirq/sim/CliffordState.md): A state of the Clifford simulation.

[`class CliffordTrialResult`](../../../cirq/sim/CliffordTrialResult.md): Results of a simulation by a SimulatesFinalState.


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.simulator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.sim.simulator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Abstract base classes for different types of simulators.


Simulator types include:

    SimulatesSamples: mimics the interface of quantum hardware.

    SimulatesAmplitudes: computes amplitudes of desired bitstrings in the
        final state of the simulation.

    SimulatesFinalState: allows access to the final state of the simulation.

    SimulatesIntermediateState: allows for access to the state of the simulation
        as the simulation iterates through the moments of a cirq.

## Classes

[`class SimulatesAmplitudes`](../../cirq/sim/SimulatesAmplitudes.md): Simulator that computes final amplitudes of given bitstrings.

[`class SimulatesFinalState`](../../cirq/sim/SimulatesFinalState.md): Simulator that allows access to the simulator's final state.

[`class SimulatesIntermediateState`](../../cirq/sim/SimulatesIntermediateState.md): A SimulatesFinalState that simulates a circuit by moments.

[`class SimulatesSamples`](../../cirq/sim/SimulatesSamples.md): Simulator that mimics running on quantum hardware.

[`class SimulationTrialResult`](../../cirq/sim/SimulationTrialResult.md): Results of a simulation by a SimulatesFinalState.

[`class StepResult`](../../cirq/sim/StepResult.md): Results of a step of a SimulatesIntermediateState.

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>

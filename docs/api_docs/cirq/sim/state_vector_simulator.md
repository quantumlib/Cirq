<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.state_vector_simulator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.sim.state_vector_simulator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Abstract classes for simulations which keep track of state vector.



## Classes

[`class SimulatesIntermediateStateVector`](../../cirq/sim/SimulatesIntermediateStateVector.md): A simulator that accesses its state vector as it does its simulation.

[`class SimulatesIntermediateWaveFunction`](../../cirq/sim/SimulatesIntermediateWaveFunction.md): Deprecated. Please use `SimulatesIntermediateStateVector` instead.

[`class StateVectorSimulatorState`](../../cirq/sim/StateVectorSimulatorState.md)

[`class StateVectorStepResult`](../../cirq/sim/StateVectorStepResult.md): Results of a step of a SimulatesIntermediateState.

[`class StateVectorTrialResult`](../../cirq/sim/StateVectorTrialResult.md): A `SimulationTrialResult` that includes the `StateVectorMixin` methods.

[`class WaveFunctionSimulatorState`](../../cirq/sim/WaveFunctionSimulatorState.md): Deprecated. Please use `StateVectorSimulatorState` instead.

[`class WaveFunctionStepResult`](../../cirq/sim/WaveFunctionStepResult.md): Deprecated. Please use `StateVectorStepResult` instead.

[`class WaveFunctionTrialResult`](../../cirq/sim/WaveFunctionTrialResult.md): Deprecated. Please use `StateVectorTrialResult` instead.

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>

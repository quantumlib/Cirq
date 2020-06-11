<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.engine.engine" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
<meta itemprop="property" content="TYPE_PREFIX"/>
</div>

# Module: cirq.google.engine.engine

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Classes for running against Google's Quantum Cloud Service.


As an example, to run a circuit against the xmon simulator on the cloud,
    engine = cirq.google.Engine(project_id='my-project-id')
    program = engine.create_program(circuit)
    result0 = program.run(params=params0, repetitions=10)
    result1 = program.run(params=params1, repetitions=10)

In order to run on must have access to the Quantum Engine API. Access to this
API is (as of June 22, 2018) restricted to invitation only.

## Modules

[`engine_client`](../../../cirq/google/engine/engine_client.md) module

[`engine_job`](../../../cirq/google/engine/engine_job.md) module: A helper for jobs that have been created on the Quantum Engine.

[`engine_processor`](../../../cirq/google/engine/engine_processor.md) module

[`engine_program`](../../../cirq/google/engine/engine_program.md) module

[`engine_sampler`](../../../cirq/google/engine/engine_sampler.md) module

## Classes

[`class Engine`](../../../cirq/google/Engine.md): Runs programs via the Quantum Engine API.

[`class EngineContext`](../../../cirq/google/engine/engine/EngineContext.md): Context for running against the Quantum Engine API. Most users should

[`class ProtoVersion`](../../../cirq/google/ProtoVersion.md): Protocol buffer version to use for requests to the quantum engine.

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
* `TYPE_PREFIX = 'type.googleapis.com/'` <a id="TYPE_PREFIX"></a>

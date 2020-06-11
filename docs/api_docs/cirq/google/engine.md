<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.engine" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.google.engine

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Client for running on Google's Quantum Engine.



## Modules

[`calibration`](../../cirq/google/engine/calibration.md) module: Calibration wrapper for calibrations returned from the Quantum Engine.

[`client`](../../cirq/google/engine/client.md) module

[`engine`](../../cirq/google/engine/engine.md) module: Classes for running against Google's Quantum Cloud Service.

[`engine_client`](../../cirq/google/engine/engine_client.md) module

[`engine_job`](../../cirq/google/engine/engine_job.md) module: A helper for jobs that have been created on the Quantum Engine.

[`engine_processor`](../../cirq/google/engine/engine_processor.md) module

[`engine_program`](../../cirq/google/engine/engine_program.md) module

[`engine_sampler`](../../cirq/google/engine/engine_sampler.md) module

[`engine_timeslot`](../../cirq/google/engine/engine_timeslot.md) module

[`env_config`](../../cirq/google/engine/env_config.md) module: Utility methods for getting configured Engine instances.

## Classes

[`class Calibration`](../../cirq/google/Calibration.md): A convenience wrapper for calibrations that acts like a dictionary.

[`class Engine`](../../cirq/google/Engine.md): Runs programs via the Quantum Engine API.

[`class EngineException`](../../cirq/google/engine/EngineException.md): Common base class for all non-exit exceptions.

[`class EngineJob`](../../cirq/google/EngineJob.md): A job created via the Quantum Engine API.

[`class EngineProcessor`](../../cirq/google/EngineProcessor.md): A processor available via the Quantum Engine API.

[`class EngineProgram`](../../cirq/google/EngineProgram.md): A program created via the Quantum Engine API.

[`class EngineTimeSlot`](../../cirq/google/EngineTimeSlot.md): A python wrapping of a Quantum Engine timeslot.

[`class ProtoVersion`](../../cirq/google/ProtoVersion.md): Protocol buffer version to use for requests to the quantum engine.

[`class QuantumEngineSampler`](../../cirq/google/QuantumEngineSampler.md): A sampler that samples from processors managed by the Quantum Engine.

## Functions

[`engine_from_environment(...)`](../../cirq/google/engine_from_environment.md): Returns an Engine instance configured using environment variables.


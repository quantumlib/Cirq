<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.Engine" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_program"/>
<meta itemprop="property" content="get_processor"/>
<meta itemprop="property" content="get_program"/>
<meta itemprop="property" content="list_processors"/>
<meta itemprop="property" content="run"/>
<meta itemprop="property" content="run_sweep"/>
<meta itemprop="property" content="sampler"/>
</div>

# cirq.google.Engine

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Runs programs via the Quantum Engine API.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.Engine`, `cirq.google.engine.engine.Engine`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.Engine(
    project_id: str,
    proto_version: Optional[<a href="../../cirq/google/ProtoVersion.md"><code>cirq.google.ProtoVersion</code></a>] = None,
    service_args: Optional[Dict] = None,
    verbose: Optional[bool] = None,
    context: Optional[<a href="../../cirq/google/engine/engine/EngineContext.md"><code>cirq.google.engine.engine.EngineContext</code></a>] = None,
    timeout: Optional[int] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

This class has methods for creating programs and jobs that execute on
Quantum Engine:
    create_program
    run
    run_sweep

Another set of methods return information about programs and jobs that
have been previously created on the Quantum Engine, as well as metadata
about available processors:
    get_program
    list_processors
    get_processor

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`project_id`
</td>
<td>
A project_id string of the Google Cloud Project to use.
API interactions will be attributed to this project and any
resources created will be owned by the project. See
https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects
</td>
</tr><tr>
<td>
`context`
</td>
<td>
Engine configuration and context to use.
</td>
</tr><tr>
<td>
`proto_version`
</td>
<td>
The version of cirq protos to use. If None, then
ProtoVersion.V2 will be used.
</td>
</tr><tr>
<td>
`service_args`
</td>
<td>
A dictionary of arguments that can be used to
configure options on the underlying client.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Suppresses stderr messages when set to False. Default is
true.
</td>
</tr><tr>
<td>
`timeout`
</td>
<td>
Timeout for polling for results, in seconds.  Default is
to never timeout.
</td>
</tr>
</table>



## Methods

<h3 id="create_program"><code>create_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_program(
    program: "cirq.Circuit",
    program_id: Optional[str] = None,
    gate_set: <a href="../../cirq/google/SerializableGateSet.md"><code>cirq.google.SerializableGateSet</code></a> = None,
    description: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> <a href="../../cirq/google/EngineProgram.md"><code>cirq.google.EngineProgram</code></a>
</code></pre>

Wraps a Circuit for use with the Quantum Engine.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The Circuit to execute.
</td>
</tr><tr>
<td>
`program_id`
</td>
<td>
A user-provided identifier for the program. This must be
unique within the Google Cloud project being used. If this
parameter is not provided, a random id of the format
'prog-################YYMMDD' will be generated, where # is
alphanumeric and YYMMDD is the current year, month, and day.
</td>
</tr><tr>
<td>
`gate_set`
</td>
<td>
The gate set used to serialize the circuit. The gate set
must be supported by the selected processor
</td>
</tr><tr>
<td>
`description`
</td>
<td>
An optional description to set on the program.
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
Optional set of labels to set on the program.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A EngineProgram for the newly created program.
</td>
</tr>

</table>



<h3 id="get_processor"><code>get_processor</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_processor(
    processor_id: str
) -> <a href="../../cirq/google/EngineProcessor.md"><code>cirq.google.EngineProcessor</code></a>
</code></pre>

Returns an EngineProcessor for a Quantum Engine processor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`processor_id`
</td>
<td>
The processor unique identifier.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A EngineProcessor for the processor.
</td>
</tr>

</table>



<h3 id="get_program"><code>get_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_program(
    program_id: str
) -> <a href="../../cirq/google/EngineProgram.md"><code>cirq.google.EngineProgram</code></a>
</code></pre>

Returns an EngineProgram for an existing Quantum Engine program.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program_id`
</td>
<td>
Unique ID of the program within the parent project.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A EngineProgram for the program.
</td>
</tr>

</table>



<h3 id="list_processors"><code>list_processors</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_processors() -> List[<a href="../../cirq/google/EngineProcessor.md"><code>cirq.google.EngineProcessor</code></a>]
</code></pre>

Returns a list of Processors that the user has visibility to in the
current Engine project. The names of these processors are used to
identify devices when scheduling jobs and gathering calibration metrics.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of EngineProcessors to access status, device and calibration
information.
</td>
</tr>

</table>



<h3 id="run"><code>run</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run(
    program: "cirq.Circuit",
    program_id: Optional[str] = None,
    job_id: Optional[str] = None,
    param_resolver: <a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a> = study.ParamResolver({}),
    repetitions: int = 1,
    processor_ids: Sequence[str] = ('xmonsim',),
    gate_set: <a href="../../cirq/google/SerializableGateSet.md"><code>cirq.google.SerializableGateSet</code></a> = None,
    program_description: Optional[str] = None,
    program_labels: Optional[Dict[str, str]] = None,
    job_description: Optional[str] = None,
    job_labels: Optional[Dict[str, str]] = None
) -> <a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>
</code></pre>

Runs the supplied Circuit via Quantum Engine.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The Circuit to execute. If a circuit is
provided, a moment by moment schedule will be used.
</td>
</tr><tr>
<td>
`program_id`
</td>
<td>
A user-provided identifier for the program. This must
be unique within the Google Cloud project being used. If this
parameter is not provided, a random id of the format
'prog-################YYMMDD' will be generated, where # is
alphanumeric and YYMMDD is the current year, month, and day.
</td>
</tr><tr>
<td>
`job_id`
</td>
<td>
Job identifier to use. If this is not provided, a random id
of the format 'job-################YYMMDD' will be generated,
where # is alphanumeric and YYMMDD is the current year, month,
and day.
</td>
</tr><tr>
<td>
`param_resolver`
</td>
<td>
Parameters to run with the program.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of repetitions to simulate.
</td>
</tr><tr>
<td>
`processor_ids`
</td>
<td>
The engine processors that should be candidates
to run the program. Only one of these will be scheduled for
execution.
</td>
</tr><tr>
<td>
`gate_set`
</td>
<td>
The gate set used to serialize the circuit. The gate set
must be supported by the selected processor.
</td>
</tr><tr>
<td>
`program_description`
</td>
<td>
An optional description to set on the program.
</td>
</tr><tr>
<td>
`program_labels`
</td>
<td>
Optional set of labels to set on the program.
</td>
</tr><tr>
<td>
`job_description`
</td>
<td>
An optional description to set on the job.
</td>
</tr><tr>
<td>
`job_labels`
</td>
<td>
Optional set of labels to set on the job.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A single TrialResult for this run.
</td>
</tr>

</table>



<h3 id="run_sweep"><code>run_sweep</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_sweep(
    program: "cirq.Circuit",
    program_id: Optional[str] = None,
    job_id: Optional[str] = None,
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a> = None,
    repetitions: int = 1,
    processor_ids: Sequence[str] = ('xmonsim',),
    gate_set: <a href="../../cirq/google/SerializableGateSet.md"><code>cirq.google.SerializableGateSet</code></a> = None,
    program_description: Optional[str] = None,
    program_labels: Optional[Dict[str, str]] = None,
    job_description: Optional[str] = None,
    job_labels: Optional[Dict[str, str]] = None
) -> <a href="../../cirq/google/EngineJob.md"><code>cirq.google.EngineJob</code></a>
</code></pre>

Runs the supplied Circuit via Quantum Engine.Creates

In contrast to run, this runs across multiple parameter sweeps, and
does not block until a result is returned.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The Circuit to execute. If a circuit is
provided, a moment by moment schedule will be used.
</td>
</tr><tr>
<td>
`program_id`
</td>
<td>
A user-provided identifier for the program. This must
be unique within the Google Cloud project being used. If this
parameter is not provided, a random id of the format
'prog-################YYMMDD' will be generated, where # is
alphanumeric and YYMMDD is the current year, month, and day.
</td>
</tr><tr>
<td>
`job_id`
</td>
<td>
Job identifier to use. If this is not provided, a random id
of the format 'job-################YYMMDD' will be generated,
where # is alphanumeric and YYMMDD is the current year, month,
and day.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
Parameters to run with the program.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of circuit repetitions to run.
</td>
</tr><tr>
<td>
`processor_ids`
</td>
<td>
The engine processors that should be candidates
to run the program. Only one of these will be scheduled for
execution.
</td>
</tr><tr>
<td>
`gate_set`
</td>
<td>
The gate set used to serialize the circuit. The gate set
must be supported by the selected processor.
</td>
</tr><tr>
<td>
`program_description`
</td>
<td>
An optional description to set on the program.
</td>
</tr><tr>
<td>
`program_labels`
</td>
<td>
Optional set of labels to set on the program.
</td>
</tr><tr>
<td>
`job_description`
</td>
<td>
An optional description to set on the job.
</td>
</tr><tr>
<td>
`job_labels`
</td>
<td>
Optional set of labels to set on the job.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An EngineJob. If this is iterated over it returns a list of
TrialResults, one for each parameter sweep.
</td>
</tr>

</table>



<h3 id="sampler"><code>sampler</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sampler(
    processor_id: Union[str, List[str]],
    gate_set: <a href="../../cirq/google/SerializableGateSet.md"><code>cirq.google.SerializableGateSet</code></a>
) -> <a href="../../cirq/google/QuantumEngineSampler.md"><code>cirq.google.QuantumEngineSampler</code></a>
</code></pre>

Returns a sampler backed by the engine.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`processor_id`
</td>
<td>
String identifier, or list of string identifiers,
determining which processors may be used when sampling.
</td>
</tr><tr>
<td>
`gate_set`
</td>
<td>
Determines how to serialize circuits when requesting
samples.
</td>
</tr>
</table>






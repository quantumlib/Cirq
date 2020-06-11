<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.EngineProgram" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_labels"/>
<meta itemprop="property" content="create_time"/>
<meta itemprop="property" content="delete"/>
<meta itemprop="property" content="description"/>
<meta itemprop="property" content="engine"/>
<meta itemprop="property" content="get_circuit"/>
<meta itemprop="property" content="get_job"/>
<meta itemprop="property" content="labels"/>
<meta itemprop="property" content="remove_labels"/>
<meta itemprop="property" content="run"/>
<meta itemprop="property" content="run_sweep"/>
<meta itemprop="property" content="set_description"/>
<meta itemprop="property" content="set_labels"/>
<meta itemprop="property" content="update_time"/>
</div>

# cirq.google.EngineProgram

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A program created via the Quantum Engine API.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.EngineProgram`, `cirq.google.engine.engine.engine_program.EngineProgram`, `cirq.google.engine.engine_program.EngineProgram`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.EngineProgram(
    project_id: str,
    program_id: str,
    context: "engine_base.EngineContext",
    _program: Optional[qtypes.QuantumProgram] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

This program wraps a Circuit with additional metadata used to
schedule against the devices managed by Quantum Engine.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`project_id`
</td>
<td>
A project_id of the parent Google Cloud Project.
</td>
</tr><tr>
<td>
`program_id`
</td>
<td>
Unique ID of the program within the parent project.
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
`_program`
</td>
<td>
The optional current program state.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`project_id`
</td>
<td>
A project_id of the parent Google Cloud Project.
</td>
</tr><tr>
<td>
`program_id`
</td>
<td>
Unique ID of the program within the parent project.
</td>
</tr>
</table>



## Methods

<h3 id="add_labels"><code>add_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_labels(
    labels: Dict[str, str]
) -> "EngineProgram"
</code></pre>

Adds new labels to a previously created quantum program.


#### Params:


* <b>`labels`</b>: New labels to add to the existing program labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineProgram.
</td>
</tr>

</table>



<h3 id="create_time"><code>create_time</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_time() -> "datetime.datetime"
</code></pre>

Returns when the program was created.


<h3 id="delete"><code>delete</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete(
    delete_jobs: bool = False
) -> None
</code></pre>

Deletes a previously created quantum program.


#### Params:


* <b>`delete_jobs`</b>: If True will delete all the program's jobs, other this
    will fail if the program contains any jobs.


<h3 id="description"><code>description</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>description() -> str
</code></pre>

Returns the description of the program.


<h3 id="engine"><code>engine</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>engine() -> "engine_base.Engine"
</code></pre>

Returns the parent Engine object.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The program's parent Engine.
</td>
</tr>

</table>



<h3 id="get_circuit"><code>get_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_circuit() -> "Circuit"
</code></pre>

Returns the cirq Circuit for the Quantum Engine program. This is only
supported if the program was created with the V2 protos.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The program's cirq Circuit.
</td>
</tr>

</table>



<h3 id="get_job"><code>get_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_job(
    job_id: str
) -> <a href="../../cirq/google/EngineJob.md"><code>cirq.google.EngineJob</code></a>
</code></pre>

Returns an EngineJob for an existing Quantum Engine job.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`job_id`
</td>
<td>
Unique ID of the job within the parent program.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A EngineJob for the job.
</td>
</tr>

</table>



<h3 id="labels"><code>labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>labels() -> Dict[str, str]
</code></pre>

Returns the labels of the program.


<h3 id="remove_labels"><code>remove_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_labels(
    keys: List[str]
) -> "EngineProgram"
</code></pre>

Removes labels with given keys from the labels of a previously
created quantum program.

#### Params:


* <b>`label_keys`</b>: Label keys to remove from the existing program labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineProgram.
</td>
</tr>

</table>



<h3 id="run"><code>run</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run(
    job_id: Optional[str] = None,
    param_resolver: <a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a> = study.ParamResolver({}),
    repetitions: int = 1,
    processor_ids: Sequence[str] = ('xmonsim',),
    description: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> <a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>
</code></pre>

Runs the supplied Circuit via Quantum Engine.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`job_id`
</td>
<td>
Optional job id to use. If this is not provided, a random id
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
`description`
</td>
<td>
An optional description to set on the job.
</td>
</tr><tr>
<td>
`labels`
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_sweep(
    job_id: Optional[str] = None,
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a> = None,
    repetitions: int = 1,
    processor_ids: Sequence[str] = ('xmonsim',),
    description: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> <a href="../../cirq/google/EngineJob.md"><code>cirq.google.EngineJob</code></a>
</code></pre>

Runs the program on the QuantumEngine.

In contrast to run, this runs across multiple parameter sweeps, and
does not block until a result is returned.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`job_id`
</td>
<td>
Optional job id to use. If this is not provided, a random id
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
`description`
</td>
<td>
An optional description to set on the job.
</td>
</tr><tr>
<td>
`labels`
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



<h3 id="set_description"><code>set_description</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_description(
    description: str
) -> "EngineProgram"
</code></pre>

Sets the description of the program.


#### Params:


* <b>`description`</b>: The new description for the program.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineProgram.
</td>
</tr>

</table>



<h3 id="set_labels"><code>set_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_labels(
    labels: Dict[str, str]
) -> "EngineProgram"
</code></pre>

Sets (overwriting) the labels for a previously created quantum
program.

#### Params:


* <b>`labels`</b>: The entire set of new program labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineProgram.
</td>
</tr>

</table>



<h3 id="update_time"><code>update_time</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_program.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_time() -> "datetime.datetime"
</code></pre>

Returns when the program was last updated.





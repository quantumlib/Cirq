<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.EngineJob" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="add_labels"/>
<meta itemprop="property" content="cancel"/>
<meta itemprop="property" content="create_time"/>
<meta itemprop="property" content="delete"/>
<meta itemprop="property" content="description"/>
<meta itemprop="property" content="engine"/>
<meta itemprop="property" content="failure"/>
<meta itemprop="property" content="get_calibration"/>
<meta itemprop="property" content="get_processor"/>
<meta itemprop="property" content="get_repetitions_and_sweeps"/>
<meta itemprop="property" content="labels"/>
<meta itemprop="property" content="processor_ids"/>
<meta itemprop="property" content="program"/>
<meta itemprop="property" content="remove_labels"/>
<meta itemprop="property" content="results"/>
<meta itemprop="property" content="set_description"/>
<meta itemprop="property" content="set_labels"/>
<meta itemprop="property" content="status"/>
<meta itemprop="property" content="update_time"/>
</div>

# cirq.google.EngineJob

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A job created via the Quantum Engine API.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.EngineJob`, `cirq.google.engine.engine.engine_job.EngineJob`, `cirq.google.engine.engine_job.EngineJob`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.EngineJob(
    project_id: str,
    program_id: str,
    job_id: str,
    context: "engine_base.EngineContext",
    _job: Optional[quantum.types.QuantumJob] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

This job may be in a variety of states. It may be scheduling, it may be
executing on a machine, or it may have entered a terminal state
(either succeeding or failing).

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
`job_id`
</td>
<td>
Unique ID of the job within the parent program.
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
`_job`
</td>
<td>
The optional current job state.
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
</tr><tr>
<td>
`job_id`
</td>
<td>
Unique ID of the job within the parent program.
</td>
</tr>
</table>



## Methods

<h3 id="add_labels"><code>add_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_labels(
    labels: Dict[str, str]
) -> "EngineJob"
</code></pre>

Adds new labels to a previously created quantum job.


#### Params:


* <b>`labels`</b>: New labels to add to the existing job labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineJob.
</td>
</tr>

</table>



<h3 id="cancel"><code>cancel</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cancel() -> None
</code></pre>

Cancel the job.


<h3 id="create_time"><code>create_time</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_time() -> "datetime.datetime"
</code></pre>

Returns when the job was created.


<h3 id="delete"><code>delete</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete() -> None
</code></pre>

Deletes the job and result, if any.


<h3 id="description"><code>description</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>description() -> str
</code></pre>

Returns the description of the job.


<h3 id="engine"><code>engine</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>engine() -> "engine_base.Engine"
</code></pre>

Returns the parent Engine object.


<h3 id="failure"><code>failure</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>failure() -> Optional[Tuple[str, str]]
</code></pre>

Return failure code and message of the job if present.


<h3 id="get_calibration"><code>get_calibration</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_calibration() -> Optional[<a href="../../cirq/google/Calibration.md"><code>cirq.google.Calibration</code></a>]
</code></pre>

Returns the recorded calibration at the time when the job was run, if
one was captured, else None.

<h3 id="get_processor"><code>get_processor</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_processor() -> "Optional[engine_processor.EngineProcessor]"
</code></pre>

Returns the EngineProcessor for the processor the job is/was run on,
if available, else None.

<h3 id="get_repetitions_and_sweeps"><code>get_repetitions_and_sweeps</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_repetitions_and_sweeps() -> Tuple[int, List[<a href="../../cirq/study/Sweep.md"><code>cirq.study.Sweep</code></a>]]
</code></pre>

Returns the repetitions and sweeps for the Quantum Engine job.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of the repetition count and list of sweeps.
</td>
</tr>

</table>



<h3 id="labels"><code>labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>labels() -> Dict[str, str]
</code></pre>

Returns the labels of the job.


<h3 id="processor_ids"><code>processor_ids</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>processor_ids() -> List[str]
</code></pre>

Returns the processor ids provided when the job was created.


<h3 id="program"><code>program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>program() -> "engine_program.EngineProgram"
</code></pre>

Returns the parent EngineProgram object.


<h3 id="remove_labels"><code>remove_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_labels(
    keys: List[str]
) -> "EngineJob"
</code></pre>

Removes labels with given keys from the labels of a previously
created quantum job.

#### Params:


* <b>`label_keys`</b>: Label keys to remove from the existing job labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineJob.
</td>
</tr>

</table>



<h3 id="results"><code>results</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>results() -> List[<a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>]
</code></pre>

Returns the job results, blocking until the job is complete.
        

<h3 id="set_description"><code>set_description</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_description(
    description: str
) -> "EngineJob"
</code></pre>

Sets the description of the job.


#### Params:


* <b>`description`</b>: The new description for the job.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineJob.
</td>
</tr>

</table>



<h3 id="set_labels"><code>set_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_labels(
    labels: Dict[str, str]
) -> "EngineJob"
</code></pre>

Sets (overwriting) the labels for a previously created quantum job.


#### Params:


* <b>`labels`</b>: The entire set of new job labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This EngineJob.
</td>
</tr>

</table>



<h3 id="status"><code>status</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>status() -> str
</code></pre>

Return the execution status of the job.


<h3 id="update_time"><code>update_time</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_time() -> "datetime.datetime"
</code></pre>

Returns when the job was last updated.


<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_job.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator[<a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>]
</code></pre>







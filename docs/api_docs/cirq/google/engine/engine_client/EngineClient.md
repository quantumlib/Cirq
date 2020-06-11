<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.engine.engine_client.EngineClient" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_job_labels"/>
<meta itemprop="property" content="add_program_labels"/>
<meta itemprop="property" content="cancel_job"/>
<meta itemprop="property" content="cancel_reservation"/>
<meta itemprop="property" content="create_job"/>
<meta itemprop="property" content="create_program"/>
<meta itemprop="property" content="create_reservation"/>
<meta itemprop="property" content="delete_job"/>
<meta itemprop="property" content="delete_program"/>
<meta itemprop="property" content="delete_reservation"/>
<meta itemprop="property" content="get_calibration"/>
<meta itemprop="property" content="get_current_calibration"/>
<meta itemprop="property" content="get_job"/>
<meta itemprop="property" content="get_job_results"/>
<meta itemprop="property" content="get_processor"/>
<meta itemprop="property" content="get_program"/>
<meta itemprop="property" content="get_reservation"/>
<meta itemprop="property" content="list_calibrations"/>
<meta itemprop="property" content="list_processors"/>
<meta itemprop="property" content="list_reservations"/>
<meta itemprop="property" content="list_time_slots"/>
<meta itemprop="property" content="remove_job_labels"/>
<meta itemprop="property" content="remove_program_labels"/>
<meta itemprop="property" content="set_job_description"/>
<meta itemprop="property" content="set_job_labels"/>
<meta itemprop="property" content="set_program_description"/>
<meta itemprop="property" content="set_program_labels"/>
<meta itemprop="property" content="update_reservation"/>
</div>

# cirq.google.engine.engine_client.EngineClient

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Client for the Quantum Engine API that deals with the engine protos and

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.engine.engine_client.EngineClient`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.engine.engine_client.EngineClient(
    service_args: Optional[Dict] = None,
    verbose: Optional[bool] = None,
    max_retry_delay_seconds: int = 3600
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
the gRPC client but not cirq protos or objects. All users are likely better
served by using the Engine, EngineProgram, EngineJob, EngineProcessor, and
Calibration objects instead of using this directly.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`service_args`
</td>
<td>
A dictionary of arguments that can be used to
configure options on the underlying gRPC client.
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
`max_retry_delay_seconds`
</td>
<td>
The maximum number of seconds to retry when
a retryable error code is returned.
</td>
</tr>
</table>



## Methods

<h3 id="add_job_labels"><code>add_job_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_job_labels(
    project_id: str,
    program_id: str,
    job_id: str,
    labels: Dict[str, str]
) -> qtypes.QuantumJob
</code></pre>

Adds new labels to a previously created quantum job.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.
* <b>`labels`</b>: New labels to add to the existing job labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum job.
</td>
</tr>

</table>



<h3 id="add_program_labels"><code>add_program_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_program_labels(
    project_id: str,
    program_id: str,
    labels: Dict[str, str]
) -> qtypes.QuantumProgram
</code></pre>

Adds new labels to a previously created quantum program.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`labels`</b>: New labels to add to the existing program labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum program.
</td>
</tr>

</table>



<h3 id="cancel_job"><code>cancel_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cancel_job(
    project_id: str,
    program_id: str,
    job_id: str
) -> None
</code></pre>

Cancels the given job.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.


<h3 id="cancel_reservation"><code>cancel_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cancel_reservation(
    project_id: str,
    processor_id: str,
    reservation_id: str
)
</code></pre>

Cancels a quantum reservation.

This action is only valid if the associated [QuantumProcessor]
schedule not been frozen. Otherwise, delete_reservation should
be used.

The reservation will be truncated to end at the time when the request is
serviced and any remaining time will be made available as an open swim
period. This action will only succeed if the reservation has not yet
ended and is within the processor's freeze window. If the reservation
has already ended or is beyond the processor's freeze window, then the
call will return an error.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`reservation_id`</b>: Unique ID of the reservation in the parent project,


<h3 id="create_job"><code>create_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_job(
    project_id: str,
    program_id: str,
    job_id: Optional[str],
    processor_ids: Sequence[str],
    run_context: qtypes.any_pb2.Any,
    priority: Optional[int] = None,
    description: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> Tuple[str, qtypes.QuantumJob]
</code></pre>

Creates and runs a job on Quantum Engine.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

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
`run_context`
</td>
<td>
Properly serialized run context.
</td>
</tr><tr>
<td>
`processor_ids`
</td>
<td>
List of processor id for running the program.
</td>
</tr><tr>
<td>
`priority`
</td>
<td>
Optional priority to run at, 0-1000.
</td>
</tr><tr>
<td>
`description`
</td>
<td>
Optional description to set on the job.
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
Tuple of created job id and job
</td>
</tr>

</table>



<h3 id="create_program"><code>create_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_program(
    project_id: str,
    program_id: Optional[str],
    code: qtypes.any_pb2.Any,
    description: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> Tuple[str, qtypes.QuantumProgram]
</code></pre>

Creates a Quantum Engine program.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

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
`code`
</td>
<td>
Properly serialized program code.
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
Tuple of created program id and program
</td>
</tr>

</table>



<h3 id="create_reservation"><code>create_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_reservation(
    project_id: str,
    processor_id: str,
    start: datetime.datetime,
    end: datetime.datetime,
    whitelisted_users: Optional[List[str]] = None
)
</code></pre>

Creates a quantum reservation and returns the created object.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`reservation_id`</b>: Unique ID of the reservation in the parent project,
    or None if the engine should generate an id
* <b>`start`</b>: the starting time of the reservation as a datetime object
* <b>`end`</b>: the ending time of the reservation as a datetime object
* <b>`whitelisted_users`</b>: a list of emails that can use the reservation.


<h3 id="delete_job"><code>delete_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete_job(
    project_id: str,
    program_id: str,
    job_id: str
) -> None
</code></pre>

Deletes a previously created quantum job.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.


<h3 id="delete_program"><code>delete_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete_program(
    project_id: str,
    program_id: str,
    delete_jobs: bool = False
) -> None
</code></pre>

Deletes a previously created quantum program.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`delete_jobs`</b>: If True will delete all the program's jobs, other this
    will fail if the program contains any jobs.


<h3 id="delete_reservation"><code>delete_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete_reservation(
    project_id: str,
    processor_id: str,
    reservation_id: str
)
</code></pre>

Deletes a quantum reservation.

This action is only valid if the associated [QuantumProcessor]
schedule has not been frozen.  Otherwise, cancel_reservation
should be used.

If the reservation has already ended or is within the processor's
freeze window, then the call will return a `FAILED_PRECONDITION` error.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`reservation_id`</b>: Unique ID of the reservation in the parent project,


<h3 id="get_calibration"><code>get_calibration</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_calibration(
    project_id: str,
    processor_id: str,
    calibration_timestamp_seconds: int
) -> qtypes.QuantumCalibration
</code></pre>

Returns a quantum calibration.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`calibration_timestamp_seconds`</b>: The timestamp of the calibration in
    seconds.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The quantum calibration.
</td>
</tr>

</table>



<h3 id="get_current_calibration"><code>get_current_calibration</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_current_calibration(
    project_id: str,
    processor_id: str
) -> Optional[qtypes.QuantumCalibration]
</code></pre>

Returns the current quantum calibration for a processor if it has
one.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The quantum calibration or None if there is no current calibration.
</td>
</tr>

</table>



<h3 id="get_job"><code>get_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_job(
    project_id: str,
    program_id: str,
    job_id: str,
    return_run_context: bool
) -> qtypes.QuantumJob
</code></pre>

Returns a previously created job.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.


<h3 id="get_job_results"><code>get_job_results</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_job_results(
    project_id: str,
    program_id: str,
    job_id: str
) -> qtypes.QuantumResult
</code></pre>

Returns the results of a completed job.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The quantum result.
</td>
</tr>

</table>



<h3 id="get_processor"><code>get_processor</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_processor(
    project_id: str,
    processor_id: str
) -> qtypes.QuantumProcessor
</code></pre>

Returns a quantum processor.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The quantum processor.
</td>
</tr>

</table>



<h3 id="get_program"><code>get_program</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_program(
    project_id: str,
    program_id: str,
    return_code: bool
) -> qtypes.QuantumProgram
</code></pre>

Returns a previously created quantum program.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`return_code`</b>: If True returns the serialized program code.


<h3 id="get_reservation"><code>get_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_reservation(
    project_id: str,
    processor_id: str,
    reservation_id: str
)
</code></pre>

Gets a quantum reservation from the engine.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`reservation_id`</b>: Unique ID of the reservation in the parent project,


<h3 id="list_calibrations"><code>list_calibrations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_calibrations(
    project_id: str,
    processor_id: str,
    filter_str: str = ''
) -> List[qtypes.QuantumCalibration]
</code></pre>

Returns a list of quantum calibrations.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`filter`</b>: Filter string current only supports 'timestamp' with values
  of epoch time in seconds or short string 'yyyy-MM-dd' or long
  string 'yyyy-MM-dd HH:mm:ss.SSS' both in UTC. For example:
    'timestamp > 1577960125 AND timestamp <= 1578241810'
    'timestamp > 2020-01-02 AND timestamp <= 2020-01-05'
    'timestamp > "2020-01-02 10:15:25.000" AND timestamp <=
      "2020-01-05 16:30:10.456"'


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of calibrations.
</td>
</tr>

</table>



<h3 id="list_processors"><code>list_processors</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_processors(
    project_id: str
) -> List[qtypes.QuantumProcessor]
</code></pre>

Returns a list of Processors that the user has visibility to in the
current Engine project. The names of these processors are used to
identify devices when scheduling jobs and gathering calibration metrics.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of metadata of each processor.
</td>
</tr>

</table>



<h3 id="list_reservations"><code>list_reservations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_reservations(
    project_id: str,
    processor_id: str,
    filter_str: str = ''
) -> List[qtypes.QuantumReservation]
</code></pre>

Returns a list of quantum reservations.

Only reservations owned by this project will be returned.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`filter`</b>: A string for filtering quantum reservations.
    The fields eligible for filtering are start_time and end_time
    Examples:
        `start_time >= 1584385200`: Reservation began on or after
            the epoch time Mar 16th, 7pm GMT.
        `end_time >= "2017-01-02 15:21:15.142"`: Reservation ends on
            or after Jan 2nd 2017 15:21:15.142


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of QuantumReservation objects.
</td>
</tr>

</table>



<h3 id="list_time_slots"><code>list_time_slots</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>list_time_slots(
    project_id: str,
    processor_id: str,
    filter_str: str = ''
) -> List[qtypes.QuantumTimeSlot]
</code></pre>

Returns a list of quantum time slots on a processor.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`filter`</b>: A string expression for filtering the quantum
    time slots returned by the list command. The fields
    eligible for filtering are `start_time`, `end_time`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of QuantumTimeSlot objects.
</td>
</tr>

</table>



<h3 id="remove_job_labels"><code>remove_job_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_job_labels(
    project_id: str,
    program_id: str,
    job_id: str,
    label_keys: List[str]
) -> qtypes.QuantumJob
</code></pre>

Removes labels with given keys from the labels of a previously
created quantum job.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.
* <b>`label_keys`</b>: Label keys to remove from the existing job labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum job.
</td>
</tr>

</table>



<h3 id="remove_program_labels"><code>remove_program_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_program_labels(
    project_id: str,
    program_id: str,
    label_keys: List[str]
) -> qtypes.QuantumProgram
</code></pre>

Removes labels with given keys from the labels of a previously
created quantum program.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`label_keys`</b>: Label keys to remove from the existing program labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum program.
</td>
</tr>

</table>



<h3 id="set_job_description"><code>set_job_description</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_job_description(
    project_id: str,
    program_id: str,
    job_id: str,
    description: str
) -> qtypes.QuantumJob
</code></pre>

Sets the description for a previously created quantum job.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.
* <b>`description`</b>: The new job description.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum job.
</td>
</tr>

</table>



<h3 id="set_job_labels"><code>set_job_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_job_labels(
    project_id: str,
    program_id: str,
    job_id: str,
    labels: Dict[str, str]
) -> qtypes.QuantumJob
</code></pre>

Sets (overwriting) the labels for a previously created quantum job.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`job_id`</b>: Unique ID of the job within the parent program.
* <b>`labels`</b>: The entire set of new job labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum job.
</td>
</tr>

</table>



<h3 id="set_program_description"><code>set_program_description</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_program_description(
    project_id: str,
    program_id: str,
    description: str
) -> qtypes.QuantumProgram
</code></pre>

Sets the description for a previously created quantum program.


#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`description`</b>: The new program description.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum program.
</td>
</tr>

</table>



<h3 id="set_program_labels"><code>set_program_labels</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_program_labels(
    project_id: str,
    program_id: str,
    labels: Dict[str, str]
) -> qtypes.QuantumProgram
</code></pre>

Sets (overwriting) the labels for a previously created quantum
program.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`program_id`</b>: Unique ID of the program within the parent project.
* <b>`labels`</b>: The entire set of new program labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The updated quantum program.
</td>
</tr>

</table>



<h3 id="update_reservation"><code>update_reservation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_client.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_reservation(
    project_id: str,
    processor_id: str,
    reservation_id: str,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    whitelisted_users: Optional[List[str]] = None
)
</code></pre>

Updates a quantum reservation.

This will update a quantum reservation's starting time, ending time,
and list of whitelisted users.  If any field is not filled, it will
not be updated.

#### Params:


* <b>`project_id`</b>: A project_id of the parent Google Cloud Project.
* <b>`processor_id`</b>: The processor unique identifier.
* <b>`reservation_id`</b>: Unique ID of the reservation in the parent project,
* <b>`start`</b>: the new starting time of the reservation as a datetime object
* <b>`end`</b>: the new ending time of the reservation as a datetime object
* <b>`whitelisted_users`</b>: a list of emails that can use the reservation.
    The empty list, [], will clear the whitelisted_users while None
    will leave the value unchanged.





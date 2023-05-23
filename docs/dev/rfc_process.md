# The Cirq RFC process

New major Cirq feature begins life as a Request for Comment (RFC).

An RFC is a document that describes a requirement and the proposed changes that will solve it. Specifically, the RFC will:

*   Be formatted according to the [RFC template](https://tinyurl.com/cirq-rfc-template) in a Google Doc
*   A github issue is submitted with the link to the Google Doc
*   Be subject to discussion and a review meeting prior to acceptance.

The purpose of a Cirq Request for Comments (RFC) is to engage the Cirq community in development, by getting feedback from stakeholders and experts, 
and communicating design changes broadly.


## What constitutes a major feature

Major features can be one of the following:
*   Features that drastically change the user experience or workflow of Cirq.
*   Features that create new library dependencies for Cirq.
*   New simulators, samplers, protocols, devices, or packages.
*   Features that drastically change performance of Cirq.
*   Features that maintainers disagree/debate about.

The following are NOT major features:

*   Fixing a bug.
*   Extending the functionality of an existing method in a natural way.

If you are not sure if a feature constitute as a “major feature”, just submit a Github issue with a description, 
and one of the maintainers will flag the issue as a major feature if necessary.

## How to submit an RFC


1. Before submitting an RFC, discuss your aims with project contributors and maintainers and get early feedback. \
Open a [feature request](https://github.com/quantumlib/Cirq/issues/new?assignees=&labels=kind%2Ffeature-request&template=feature_request.md&title=) \
and have a discussion with the maintainers. Mention that you are willing to write an RFC.
2. [Join the cirq-dev Google Group](https://groups.google.com/forum/#!forum/cirq-dev) to get an invitation to our weekly Cirq Cynq meeting. 
3. Draft your RFC. 
    * Follow the [RFC template](https://tinyurl.com/cirq-rfc-template), link the Github issue in your RFC.
    * Make sure to share your doc with cirq-dev@googlegroups.com for comments.
    * Link the RFC in your issue.
4. Recruiting a sponsor:
    * A sponsor must be a maintainer of the project or the product manager.
    * Write a comment in your Github issue that calls out that you are "Looking for a sponsor". A maintainer will mark the issue with a label: "rfc/needs-sponsor".
    * While it might take some time to get a maintainer to sponsor your RFC, it is essential, as the sponsor will facilitate the process for reviewing your design.
    * Tips to recruit a sponsor: 1) keep commenting on the issue weekly 2) attend Cirq Cynq and push for a sponsor.
5. Agree with your sponsor on a Cirq Cync meeting to present the RFC so that other contributors and maintainers can become more familiar with your design.
6. The meeting may approve the RFC, reject it, or require changes before it can be considered again. \
It is likely that multiple iterations of comments and meetings with your sponsor will be required before approval.


## RFC participants

Many people are involved in the RFC process:

*   **RFC author** — one or more community members who write an RFC and are committed to championing it through the process
*   **RFC sponsor** — a maintainer or product manager who sponsors the RFC and will shepherd it through the RFC review process
*   **review committee** — a group of maintainers who have the responsibility of recommending the adoption of the RFC.  This group can be emailed by emailing cirq-maintainers@googlegroups.com. This is typically the attendees at the Cirq Cync meeting.
*   Any **community member** may help by providing feedback on whether the RFC will meet their needs.

### RFC sponsors

A sponsor is a project maintainer responsible for ensuring the best possible outcome of the RFC process. This includes:

*   Advocating for the proposed design.
*   Guiding the review committee to come to a productive consensus.
*   If the RFC moves to implementation:
    *   Ensuring proposed implementation adheres to the design.
    *   Coordinate with appropriate parties to successfully land implementation.


### RFC review committees

The review committee decides on a consensus basis whether to approve, reject, or request changes. They are responsible for:

*   Ensuring that substantive items of public feedback have been accounted for.
*   Adding their meeting notes as comments to the PR.
*   Providing reasons for their decisions.

The constitution of a review committee may change according to the particular governance style and leadership of each project.

## Implementing new features

Once an RFC has been approved, implementation can begin.

If you are working on new code to implement an RFC:

*   Make sure you understand the feature and the design approved in the RFC. Ask questions and discuss the approach before beginning work.
*   New features must include new unit tests that verify the feature works as expected.
*   Add or update relevant API documentation. Reference the RFC in the new documentation.
*   Follow any other guidelines laid out in the [CONTRIBUTING.md](https://github.com/quantumlib/Cirq/blob/master/CONTRIBUTING.md) file in the project repo you're contributing to.
*   Run unit tests before submitting your code.
*   Work with the RFC sponsor to successfully land the new code. This could include PR / marketing of the new feature as well.
*   Follow the Cirq process of deprecating code. Please refer to [\_compat.py](https://github.com/quantumlib/Cirq/blob/master/cirq-core/cirq/_compat.py) as an example.


## Keeping the bar high

While we encourage and celebrate every contributor, the bar for RFC acceptance is kept intentionally high. A new feature may be rejected or need significant revision at any one of these stages:

*   Initial design conversations on the relevant mailing list.
*   Failure to recruit a sponsor.
*   Critical objections during the feedback phase.
*   Failure to achieve consensus during the design review.
*   Concerns raised during implementation (for example: inability to achieve backwards compatibility, concerns about maintenance).

If this process is functioning well, RFCs are expected to fail in the earlier, rather than later stages.
An approved RFC is not a commitment to implementation on any sort of timeline. The prioritization of features depends on user interest and willingness of contributors to implement them.

## New hardware integrations

Several hardware vendors already have integrations with cirq.
If you are considering integrating with cirq, please engage with us through the
cirq-maintainer team by attending the weekly cirq cync meeting and submitting
an RFC to get feedback on your design proposal (as specified above). The goal of the RFC
review process is to ensure all integrations end up with a well-maintained, user
friendly interface with a high reliability.

Examples of other integrations can be found on the
[Hardware page](../hardware).

There are a range of possibilities for integrating with cirq, including:

*    Completely independent repository: requires the least engagement from
the cirq team, but also lacks the benefits of a tighter integration and
partnership.
*    Different repository with links or tutorials hosted on cirq:
less coordination is needed than a hosted integration, but requires
continuous integration on the external repository to ensure compatibility
with new cirq versions.
*    Integration hosted within cirq repository:  requires the highest
amount of coordination and effort but allows a tighter integration and for
cirq-maintainers to modify the integration code to stay in sync with an
evolving cirq-core code base.  We generally do not accept submissions for
this type of integration for third-party vendors that function as
intermediaries for other cloud vendors.

Several things are needed for a successful integration:

*    Plan for including external dependencies (if needed).
*    Access and authentication.
*    Tutorials and guides for use of the interface (for instance,
a Getting Started guide).
*    `Device` implementation for validating circuits on the hardware
*     Transformer for compiling circuits to the supported gates on the
hardware (or use `cirq.optimize_for_target_gateset`)
*    ` Sampler` interface for running circuits on the hardware service.
*     Conisder also providing a noise model that users can use to simulate the
device if direct access is not available.



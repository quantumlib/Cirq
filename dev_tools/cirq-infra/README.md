# Cirq infra

_Cirq-infra_ is the name of a the Google Cloud Platform (GCP) project implemented and managed by the
Google Quantum AI team in support of open-source projects like Cirq. It was originally introduced
around the years 2020–2021 to implement several systems:

*   Automatic merging of pull requests on GitHub
*   Automatic labeling of the sizes of changes in pull requests
*   Running [Triage Party](https://github.com/google/triage-party), a tool for collaborative
    triaging of GitHub issues and pull requests

The automerge and automatic labeling capabilities were implemented using a homegrown tool called
`pr_monitor`, whose source code was in the [`dev_tools`](../) subdirectory of the Cirq
repository. In 2025, the automerge system was discontinued in favor of using GitHub's "merge
queues", and the size-labeling facility was reimplemented using a GitHub Actions workflow.
`pr_monitor` was retired in the Cirq 1.5 release.

_Cirq-infra_ is still being used to support Triage Party.

## GCP Configuration

Access is granted to Cirq maintainers only.

### Requirements

The tools below are required to manage our infra:

*   Cloud SDK: https://cloud.google.com/sdk/docs/quickstart
*   kubectl: `gcloud components install kubectl`.
*   skaffold: `gcloud components install skaffold`.

### GKE Cluster

We have a 3 node GKE cluster called cirq-infra. To connect to it using kubectl
use:

```shell
gcloud container clusters get-credentials cirq-infra --zone us-central1-a --project cirq-infra
```

Note that we have Workload Identity setup so that in order to access Cloud APIs
from workloads, you'll have to add permissions to the
gke-service-account@cirq-infra.iam.gserviceaccount.com service account.
Currently, it is able to access Secrets, and no other APIs.

### Secret manager

It is important to know that the Github API key in case of the Kubernetes
deployment is stored in the Cloud Secret Manager under `cirq-bot-api-key`.

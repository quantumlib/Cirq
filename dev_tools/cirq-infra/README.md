# Cirq infra 

This doc describes cirq-infra, the GCP project supporting our open source project.
The following things are planned to be running on GCP: 

- [X] Cirq bot / PR monitor - see [../pr_monitor](../pr_monitor/README.md)
- [X] Triage party for triaging 
- [ ] Performance tests and reports  

Access is granted to Cirq maintainers only.

## GCP Configuration 

### Requirements

The tools below are required to manage our infra:

- Cloud SDK: https://cloud.google.com/sdk/docs/quickstart
- kubectl: `gcloud components install kubectl`
- skaffold: `gcloud components install skaffold`


### GKE Cluster 

We have a 3 node GKE cluster called cirq-infra.
To connect to it using kubectl use:  

```
 gcloud container clusters get-credentials cirq-infra --zone us-central1-a --project cirq-infra
```

Note that we have Workload Identity setup so that in order to access Cloud APIs from workloads, \
you'll have to add permissions to the gke-service-account@cirq-infra.iam.gserviceaccount.com \
service account. Currently, it is able to access Secrets, and no other APIs.

### Secret manager 

It is important to know that the Github API key in case of the Kubernetes deployment is stored \
in the Cloud Secret Manager under `cirq-bot-api-key`. 

# PR monitor bot

The pr monitor bot continuously watches a github repository for PRs labelled with the
'automerge' label. When it sees such a PR it will mark it by labelling it with
the 'front_of_queue_automerge' label. While there is an 'front_of_queue_automerge'
labelled PR, the script will not label any other PRs with 'front_of_queue_automerge'.
If there are multiple 'automerge' PRs, the bot prefers PRs that require less work 
(e.g. tests already passed) and then breaks ties using the lowest PR number.

While there is a 'front_of_queue_automerge' labelled PR, depending on the state 
of the PR, the script might do all of the following: 
 * sync that PR with master
 * wait for status checks to succeed
 * attempt to merge it into master.
If the PR goes out of date due to an intervening merge, the process will start over.
This will continue until either the PR is merged or there is a problem that must be
addressed by a human. After merging, the PR will be deleted unless it belongs to a
fork.

The commit message used when merging a PR is the title of the PR and then, for details,
the body of the PR's initial message/comment. Users/admins should edit the title and
initial comment to appropriately describe the PR.

This script will also automatically label PRs based on their code size. The following
labels are given based on the total change numbers (addition + deletions on github)
at the time the PR is opened (not updated as changes are made):
Extra Small (XS): < 10 total changes.
Small (S):        < 50 total changes.
Medium (M):       < 250 total changes.
Large (L):        < 1000 total changes.
Extra Large (XL): >= 1000 total changes.

## Automated Deployment Flow

The bot lives in the cirq-infra project and is deployed as a GKE Deployment \
(see [Cirq infra](../cirq-infra/README.md) for more details on our GCP setup).
 
On every push to master, Cloud Build triggers the execution of cloudbuild-deploy.yaml.

## Configuration files

### Dockerfile

The [Dockerfile](Dockerfile) in this directory simply exposes the pr_monitor.py script.
In order to be able to pull and push to our repo, you'll need to configure your Docker daemon:

```
gcloud beta auth configure-docker us-docker.pkg.dev
``` 

### Kubernetes deployment

The [statefulset.yaml](statefulset.yaml) ensures that the container survives node failures, i.e 
even if there are errors, it will restart the container. As it is a SatefulSet set to 1 replica, 
it also ensures that there is only a single instance of the application running even during 
upgrades. 

To access our cluster with kubectl use

```
gcloud container clusters get-credentials cirq-infra --zone us-central1-a
```

### Cloud Build file

The [cloudbuild-deploy.yaml](cloudbuild-deploy.yaml) describes the workflow that is executed \
when we push something to master. It is responsible for building the docker image and deploying \
the new version of the pr monitor script to GKE. 


### Skaffold file

Skaffold is a tool to simplify K8s development / deployment. 
It takes care of building an image and deploying it to a cluster in a single step 
while updating the Kubernetes manifests on the fly to point to the latest version 
of the image.

Skaffold also makes sure that the deployment is always tied to a specific image digest which makes it
easier to trace back issues. 

```
...
Step #1: Tags used in deployment:
Step #1:  - us-docker.pkg.dev/cirq-infra/cirq/pr_monitor -> us-docker.pkg.dev/cirq-infra/cirq/pr_monitor:latest@sha256:c3b5751b7af77f2ec13189ee185eafd4a4fd7fbe762bf3a081f11b43c7c63354
Step #1: Starting deploy...
...
```


## Testing the deployment before submitting

As Cirqbot is currently pretty low traffic, it is not too disruptive to just test things directly in prod.
From the root of the Cirq repo run the following command to test the deployment flow: 

```
gcloud builds submit --config dev_tools/pr_monitor/cloudbuild-deploy.yaml --project cirq-infra
```

If you want to iterate faster, you can also try using skaffold itself in dev mode: 

```
skaffold dev -f dev_tools/pr_monitor/skaffold.yaml 
```

 
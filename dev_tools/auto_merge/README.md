# Automerge bot

The automerge bot continuously watches a github repository for PRs labelled with the
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

## Automated Deployment Flow

The bot lives in the cirq-infra project and is deployed as a GKE Deployment \
(see [Cirq infra](../cirq-infra/README.md) for more details on our GCP setup).
 
On every push to master, Cloud Build triggers the execution of cloudbuild-deploy.yaml.

## Configuration files

### Dockerfile

The [Dockerfile](Dockerfile) in this directory simply exposes the auto_merge.py script. 

### Kubernetes deployment

The [deployment.yaml](deployment.yaml) ensures that the container survives node failures, i.e 
even if there are errors, it will restart the container. 

### Cloud Build file

The [cloudbuild-deploy.yaml](cloudbuild-deploy.yaml) describes the workflow that is executed \
when we push something to master. It is responsible for building the docker image and deploying \
the new version of the automerge script to GKE.  

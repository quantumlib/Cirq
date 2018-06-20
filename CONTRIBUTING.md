# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Pull Request Process and Code Review

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. 
[GitHub Help](https://help.github.com/articles/about-pull-requests/) has 
information on using pull requests.

The preferred manner for submitting pull requests is for users to fork
the Cirq [repo](https://github.com/quantumlib/Cirq) and then use a 
branch from this fork to create a pull request to the main Cirq repo.

The basic process for setting up a fork is 
1. Fork the Cirq repo (Fork button in upper right corner of 
[repo page](https://github.com/quantumlib/Cirq)).
Forking creates a new github repo at the location 
```https://github.com/USERNAME/cirq``` where ```USERNAME``` is
your github id.
1. Clone the fork you created to your local machine at the directory
where you would like to store your local copy of the code.
    ```shell
    git clone git@github.com:USERNAME/cirq.git
    ```
1. Add a remote called ```upstream``` to git.  This remote will represent
the main git repo for cirq (as opposed to the clone, which you just
created, which will be the ```origin``` remote).  This remote can be used
to sync your local git repos with the main git repo for cirq.
    ```shell
    git remote add upstream https://github.com/quantumlib/cirq.git
    ```
    To verify the remote run ```git remote -v``` and you should see both 
    the ```origin``` and ```upstream``` remotes.
1. Sync up your local git with the ```upstream``` remote:
    ```shell
    git fetch upstream
    ```
    You can check the branches that are on the ```upstream``` remote by 
    running ```git remote -va```.
Most importantly you should see ```upstream/master``` listed.
1. Merge the upstream master into your local master so that 
it is up to date
    ```shell
    git checkout master
    git merge upstream/master
    ```
    At this point your local git master should be synced with the master
    from the main cirq repo. 

The process of doing work which you would like to contribute is
then to
1. Checkout master and create a new branch from this master
    ```shell
    git checkout master -b new_branch_name
    ```
    where ```new_branch_name``` is the name of your new branch.
1. Do your work and commit your changes to this branch.
1. If you have drifted out of sync with the master from the
main cirq repo you may need to merge in changes.  To do this,
first update your local master and then merge the local master
into your branch:
    ```shell
    # Update your local master.
    git fetch upstream
    git checkout master
    git merge upstream/master
    # Merge local master into your branch.
    git checkout new_branch_name
    git merge master
    ```
    You may need to fix merge conflicts for both of these merge
    commands.
1. Finally, push your change to your clone
    ```shell
    git push origin new_branch_name
    ```
1. Now when you navigate to the cirq page on github,
[https://github.com/quantumlib/cirq](https://github.com/quantumlib/cirq)
you should see the option to create a new pull request from
your clone repository.  Alternatively you can create the pull request
by navigating to the "Pull requests" tab in the page, and selecting
the appropriate branches. 
1. The reviewer will comment on your code and may ask for changes,
you can perform these locally, and then push the new commit following
the same process as above.

## Continuous Testing and Standards

As part of creating a pull request we run checks on the pull request 
to ensure that it does not break Cirq and upholds coding standards
for the code base. 

Our guidelines are 
* All submitted code should be tested. We currently us a continuous
testing framework which runs all of tests in the cirq and docs 
directories. We also perform a coverage check which will determine
if new code or modified code is covered by the tests. We currently
will block if the tests do not pass or the coverage of new/modified
lines goes down.  For the latter there may be exceptions for ignoring
uncovered lines, please work with your reviewer on this.
* We also enforce linting (style) standards for python using pylint.
We do not enforce all linting, for a list of what lint we check,
see [.pylintrc](continuous-integration/.pylintrc).
* We use typing annotations and check these using
[mypy](http://mypy-lang.org/).
* The code in the Cirq repo is written to be compatible with
Python 3.5.  However we also support a version compatible with
Python 2.7.  We use [3to2](https://pypi.org/project/3to2/) to
perform this conversion, and we run tests against the Python 2.7
version of the code using this conversion.  In essence this means
we require that the 3.5 code needs to be convertible to 2.7 using
3to2.

Our continuous integration framework will run checks for tests, 
coverage, and lint for every pull request.  It is best practice to 
run these tests yourself before submitting a pull request.  You can 
do this by running from a shell our checker on your local changes
```shell
bash continous-integration/check.sh
```
Note that this command can run only a subset of the checks using the
```--only``` flag.  This flag value can be ``pylint``,```typecheck```,
```pytest```, ```pytest2```, or ```incremental-coverage```.

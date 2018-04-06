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

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

Our code reviews currently require a maintainer to run tests and other checks
against your pull request.
This manual continuous integration system will be in place until Cirq is
released publicly, at which point we will switch to an automated system
(e.g Travis-CI).

To ensure that the continuous integration checks will pass, you can run them
locally.
To do this, you must first install `protobuf-compiler` and `virtualenv`:

```bash
sudo apt-get install protobuf-compiler virtualenv
```

Next, from the root directory of your clone of cirq's repository, run the
continuous integration scripts:

```bash
# run linting and unit tests against local code
bash continuous-integration/pylint-pull-request.sh
bash continuous-integration/test-pull-request.sh
```

These scripts will test and lint your local changes to the code.
If you wish to run the scripts against a pull request on cirq's github
repository, you can pass in the pull-request number as an argument to the
scripts:

```bash
# download a temporary copy of pull request #214 and run tests against it
bash continuous-integration/test-pull-request.sh 214
```

Note that these are the same scripts run by maintainers in order to set the
pass/fail status indicators on github.
If you aren't a maintainer (or are but don't provide an access token argument)
the status-setting code will simply be skipped.

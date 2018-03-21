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

Our code reviews also (currently) require the reviewer to run tests for
your pull request.  To insure that these tests pass, you should run
these tests locally. To do this, you must first install the protobuf
compiler and the virtual environment:
```bash
sudo apt-get install protobuf-compiler virtualenv
```
Next step is setting Cirq home directory. Assuming that you are in the
root directory of Cirq:
```bash
CIRQ_HOME="`pwd`"
```
Then you can run the following, which assumes you are in the directory
where your changes are made:
```bash
${CIRQ_HOME}/continuous_integration/pylint-pull-request.sh
${CIRQ_HOME}/continuous_integration/test-pull-request.sh
```
Reviewers will run these tests before your code is submitted to ensure
that the tests are not broken.  This ad hoc system is in place until
Cirq is released publically when a continuous testing system will
be put in place.

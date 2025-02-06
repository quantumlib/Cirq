# How to Contribute

We'd love to accept your patches and contributions to this project. We do have
some guidelines to follow, covered in this document, but don't worry about –
or expect to – get everything right the first time! Create a pull request
(discussed below) and we'll nudge you in the right direction. Please also note
that we have a [code of conduct](CODE_OF_CONDUCT.md) to make Cirq an open and
welcoming community environment.

## Contributor License Agreement

Contributions to this project must be accompanied by a [Contributor License
Agreement](https://cla.developers.google.com/about) (CLA). You
(or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to https://cla.developers.google.com/ to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Pull Request Process and Code Review

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose.
[GitHub Help](https://help.github.com/articles/about-pull-requests/) has
information on using pull requests.

The preferred manner for submitting pull requests is for developers to fork
the Cirq [repository](https://github.com/quantumlib/Cirq) and then use a [git
branch](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
from this fork to create a pull request to the main Cirq repo. The basic
process for setting up a fork is as follows:

1.  Fork the Cirq repository (you can use the _Fork_ button in upper right
    corner of the [repository page](https://github.com/quantumlib/Cirq)).
    Forking creates a new GitHub repo at the location
    `https://github.com/USERNAME/Cirq`, where `USERNAME` is
    your GitHub user name. Use the instructions on the Cirq
    [development page](docs/dev/development.md) to download a copy to
    your local machine. You need only do this once.

1.  Check out the `main` branch and create a new branch from `main`:

    ```shell
    git checkout main -b new_branch_name
    ```

    where `new_branch_name` is the name of your new branch.

1.  Do your work and commit your changes to this branch.

1.  If your local copy has drifted out of sync with the `main` branch of the
    main Cirq repo, you may need to merge in the latest changes into your
    branch.  To do this, first update your local `main` and then merge your
    local `main` into your branch:

    ```shell
    # Track the upstream repo (if your local repo hasn't):
    git remote add upstream https://github.com/quantumlib/Cirq.git

    # Update your local main.
    git fetch upstream
    git checkout main
    git merge upstream/main
    # Merge local main into your branch.
    git checkout new_branch_name
    git merge main
    ```

    You may need to fix [merge conflicts](
    https://docs.github.com/articles/about-merge-conflicts)
    during one or both of these merge processes.

1.  Finally, push your changes to your forked copy of the Cirq repo on GitHub:

    ```shell
    git push origin new_branch_name
    ```

1.  Now when you navigate to the Cirq repository on GitHub
    (https://github.com/quantumlib/Cirq), you should see the option to create a
    new pull request from your clone repository.  Alternatively, you can create
    the pull request by navigating to the "Pull requests" tab near the top of
    the page, and selecting the appropriate branches.

1.  A reviewer will comment on your code and may ask for changes. You can
    perform the necessary changes locally, and then push the new commit
    following the same process as above.

## Development Environment Setup

Please refer to our [development page](docs/dev/development.md) for
instructions on setting up your local development environment.

## Code Testing Standards

When a pull request is created or updated, various automatic checks will
run on GitHub to ensure that the changes won't break Cirq, as well as to make
sure they meet the Cirq project's coding standards.

Cirq includes a continuous integration tool to perform testing.  See our
[development page](docs/dev/development.md) on how to run the continuous
integration checks locally.

Please be aware of the following coding standards that will be applied to any
new changes.

### Tests

Existing tests must continue to pass (or be updated) when new changes are
introduced. We use [pytest](https://docs.pytest.org) to run our
tests.

### Coverage

Code should be covered by tests. We use
[pytest-cov](https://pytest-cov.readthedocs.io) to compute coverage, and custom
tooling to filter down the output to only include new or changed code. We don't
require 100% coverage, but any uncovered code must be annotated with `# pragma:
no cover`. To ignore coverage of a single line, place `# pragma: no cover` at
the end of the line. To ignore coverage for an entire block, start the block
with a `# pragma: no cover` comment on its own line.

### Lint

Code should meet common style standards for Python and be free of error-prone
constructs. We use [Pylint](https://www.pylint.org/) to check for code lint.
To see which lint checks we enforce, see the
[dev_tools/conf/.pylintrc](dev_tools/conf/.pylintrc) file. When Pylint produces
a false positive, it can be silenced with annotations. For example, the
annotation `# pylint: disable=unused-import` would silence a warning about
an unused import.

### Types

Code should have [type annotations](https://www.python.org/dev/peps/pep-0484/).
We use [mypy](http://mypy-lang.org/) to check that type annotations are correct.
When type checking produces a false positive, it can be silenced with
annotations such as `# type: ignore`.

## Request For Comment Process for New Major Features

For larger contributions that will benefit from design reviews, please use the Cirq
[Request for Comment](docs/dev/rfc_process.md) (RFC) process.

## Developing notebooks

Please refer to our [notebooks guide](docs/dev/notebooks.md) on how to develop
iPython notebooks for documentation.

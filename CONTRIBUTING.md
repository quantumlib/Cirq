# How to Contribute

We'd love to accept your patches and contributions to this project.
We do have some guidelines to follow, covered in this document, but don't worry about (or expect to) get everything right the first time!
Create a pull request and we'll nudge you in the right direction.

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
your github id. Use the directions on the
[development page](docs/dev/development.md) to download a copy to
your local machine. You need only do this once.
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

## Code Testing Standards

When a pull request is created or updated, various automatic checks will run to ensure that the change won't break Cirq and meets our coding standards.

Cirq contains a continuous integration tool to verify testing.  See our
[development page](docs/dev/development.md) on how to run the continuous
integration checks locally.

Please be aware of the following code standards that will be applied to any
new changes.

- **Tests**.
Existing tests must continue to pass (or be updated) when new changes are introduced.
We use [pytest](https://docs.pytest.org/en/latest/) to run our tests.
- **Coverage**.
Code should be covered by tests.
We use [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) to compute coverage, and custom tooling to filter down the output to only include new or changed code.
We don't require 100% coverage, but any uncovered code must be annotated with `# coverage: ignore`.
To ignore coverage of a single line, place `# coverage: ignore` at the end of the line.
To ignore coverage for an entire block, start the block with a `# coverage: ignore` comment on its own line.
- **Lint**.
Code should meet common style standards for python and be free of error-prone constructs.
We use [pylint](https://www.pylint.org/) to check for lint.
To see which lint checks we enforce, see the [dev_tools/conf/.pylintrc](dev_tools/conf/.pylintrc) file.
When pylint produces a false positive, it can be squashed with annotations like `# pylint: disable=unused-import`.
- **Types**.
Code should have [type annotations](https://www.python.org/dev/peps/pep-0484/).
We use [mypy](http://mypy-lang.org/) to check that type annotations are correct.
When type checking produces a false positive, it can be ignored with annotations like `# type: ignore`.

## Request For Comment Process for New Major Features

For larger contributions that will benefit from design reviews, please use the [Request for Comment](https://github.com/quantumlib/cirq/docs/contributions/rfc_process.md) process.

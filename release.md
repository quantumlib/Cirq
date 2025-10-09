# Versioning and Releases

This documents our versioning approach for Cirq and how we produce new Cirq
releases.

Note that Cirq development takes place on the `main` branch in GitHub. If you
want to use a more stable version of Cirq, you should use one of the
[releases](https://github.com/quantumlib/Cirq/releases) or install the package
from PyPI using `pip install cirq`. The release from the latest commit to `main`
can be installed with `pip install --upgrade cirq~=1.0.dev`.

## Versioning

We follow the [Semantic Versioning 2.0.0](https://semver.org/) approach for
labeling Cirq releases. Each stable release is labeled with an identifier
written in the form MAJOR.MINOR.PATCH, where MAJOR, MINOR, and PATCH are
numbers. The following guarantees are provided:

1.  All packages released at the same time from the Cirq repository will share
    the same [semantic versioning](https://semver.org/) version number. (If it
    ever becomes necessary to allow packages to have different version numbers,
    this policy will be updated.)

2.  Libraries in the `cirq-core` directory (with the exception of
    `cirq-core/cirq/contrib`) adhere to the guarantees outlined in the Semantic
    Versioning specification. In summary:

    *   Bug fixes that do not affect the API will only involve incrementing the
        PATCH version.

    *   Additions and/or changes that affect the API but do so in a
        backwards-compatible way will involve incrementing the MINOR version.

    *   Additions and/or changes that affect the API in a
        backwards-_incompatible_ way will increment the MAJOR version.

3.  The contrib directory (at `cirq-core/cirq/contrib`) currently follows
    Semantic Versioning except for the MINOR version increment policy: releases
    with MINOR version increments may contain backward-incompatible
    functionality changes to its public API. (They may be changed to strictly
    follow Semantic Versioning in the future, at which point this policy will
    be updated.)

4.  Cirq vendor directories (`cirq-aqt`, `cirq-google`, `cirq-ionq`, etc.) follow
    Semantic Versioning except the MINOR version increment policy: each vendor
    directory has a separate policy on whether MINOR version increments provide
    backward-compatibility guarantees, as described in `version_policy.md` in the
    respective directory.

    1.  If `version_policy.md` does not exist in a particular vendor directory,
        MINOR version increments may contain backward-incompatible functionality
        changes to its public API.

    1.  For each vendor directory, version policies may be modified to strictly
        follow Semantic Versioning in the future.

5.  Versions based on unreleased branches of `main` will be suffixed with `.dev0`.

The rules for version changes are:

*   Increment the PATCH version if all changes are bug fixes only.

*   Increment the MINOR version if changes contain functionalities which are
    backward-compatible, or if a vendor directory or `contrib` contains
    backward-incompatible changes and the policy for the directory allows
    backward-incompatible changes for a minor version increment.

At this time, a major version increment process has not been established. Until
then, backward-incompatible changes are not allowed for `cirq-core` and vendor
directories that prohibit them for a minor version increment.

## Releases

We use GitHub's release system for creating releases.  Release are listed
[on the Cirq release page](https://github.com/quantumlib/Cirq/releases).

Our development process uses the branch named `main` for development. This
branch will always use the next unreleased minor version number with the suffix
of `.dev0`. When a release is performed, the `.dev0` will be removed and tagged
in a release branch with a version tag (vX.X.X). Then, `main` will be updated
to the next minor version. The version number of `main` can always be found in
the [version file](./cirq-core/cirq/_version.py).

### Release Schedule

Releases are made approximately every quarter (i.e., every 3 months). All Cirq
packages (including vendor packages such as `cirq-aqt`) are released at the
same time.

## Before you release: flush the deprecation backlog

Ensure that all the deprecations are removed that were meant to be deprecated
for the given release. E.g. if you want to release `v0.11`, you can check with
`git grep 'v0.11'` for all the lines containing this deadline. Make sure none
of those are released.

## Release Procedure

This procedure can be followed by authorized Cirq developers to perform a
release.

### Preparation

System requirements: Linux, Python 3.11.

For MINOR/MAJOR releases: make sure you're on an up-to-date `main` branch and
in Cirq's root directory.

```bash
git checkout main
git pull origin main  # or upstream main
git status  # should be no pending changes
```

For PATCH update: Make sure you checked out the version you want to patch.
Typically this will be something like `${MAJOR}.${MINOR}.${LAST_PATCH}`

```bash
git fetch origin # or upstream - to fetch all tags
git checkout <desired tag to patch>
git status  # should be no pending changes
```

Ensure you have PyPI and Test PyPI accounts with access to the Cirq
distribution. This can be done by visiting https://test.pypi.org, logging in,
and accessing the https://test.pypi.org/project/cirq page.

For the following script to work, you will need the following environment
variables defined: `CIRQ_TEST_PYPI_TOKEN`, `CIRQ_PYPI_TOKEN`.

Also define these variables for the versions you are releasing:

```bash
VER=VERSION_YOU_WANT_TO_RELEASE  # e.g. "0.7.0"
NEXT_VER=NEXT_VERSION  # e.g. "0.8.0" (skip for PATCH releases)
```

### Create a release branch

Create a release branch called "v${VER}-dev":

```bash
git checkout -b "v${VER}-dev"
```

If you are doing a PATCH update, use `git cherry-pick` to integrate the commits
for the fixes you want to include in your update, making sure to resolve all
merge conflicts carefully:

```bash
git cherry-pick <commit>
```

Bump the version number on the release branch:

```bash
python dev_tools/modules.py replace_version --old ${VER}.dev0 --new ${VER}
git add .
git commit -m "Removing ${VER}.dev0 -> ${VER}"
git push origin "v${VER}-dev"
```

### Bump the main version

WARNING: Only bump the `main` version for minor and major releases. For PATCH
updates, leave it as it is.

```bash
git checkout main -b "version_bump_${NEXT_VER}"
python dev_tools/modules.py replace_version --old ${VER}.dev0 --new ${NEXT_VER}.dev0
git add .
git commit -m "Bump cirq version to ${NEXT_VER}"
git push origin "version_bump_${NEXT_VER}"
```

The `main` branch should never see a non-dev version specifier.

### Create the distribution wheel

From a release branch, create a binary distribution wheel. This is the package
that will go to PyPI.

```bash
git checkout "v${VER}-dev"
./dev_tools/packaging/produce-package.sh dist
ls dist  # should only contain one file, for each modules
```

### Push to Test PyPI

The package server PyPI has a [test server](https://test.pypi.org) where
packages can be uploaded to check that they work correctly before pushing the
real version. This section illustrates how to upload the package to Test PyPI
and verify that it works.

First, upload the package in the `dist/` directory. (Ensure that this is the
only package in this directory, or modify the commands to upload only this
file).

```bash
twine upload --repository-url=https://test.pypi.org/legacy/ \
  --password="$CIRQ_TEST_PYPI_TOKEN" "dist/*"
```

Next, run automated verification. Note: sometimes the first verification from
Test PyPI will fail:

```bash
# NOTE: FIRST RUN MAY FAIL - PyPI might not have indexed the version yet
./dev_tools/packaging/verify-published-package.sh "${VER}" --test
```

Once this runs, you can create a virtual environment to perform
manual verification as a sanity check and to check version number and
any high-risk features that have changed this release.

```bash
mkvirtualenv "verify_test_${VER}" --python=/usr/bin/python3
pip install -r dev_tools/requirements/dev.env.txt
pip install --extra-index-url=https://test.pypi.org/simple/ cirq=="${VER}"
python -c "import cirq; print(cirq.__version__)"
python  # just do some stuff checking that latest features are present
```

### Draft release notes and email

Put together a release notes document that can be used as part of a
release and for an announcement email.

You can model the release notes on the previous release from the
[Release page](https://github.com/quantumlib/Cirq/releases).

1.  Fill out the new version in "Tag Version" and choose your release
    branch to create the tag from.

2.  Attach the generated `.whl` file to the release.

Retrieve all commits since the last release with:

```shell
git log "--pretty=%h %s"
```

You can get the changes to the top-level objects and protocols by
checking the history of the init files.

```shell
git diff <previous version>..HEAD cirq-core/cirq/__init__.py
```

You can get the contributing authors for the release by running:

```shell
git log <previous version>..HEAD --pretty="%an" | sort |\
  uniq | sed ':a;N;$!ba;s/\n/, /g'
```

### Release to production PyPI

Upload to prod PyPI using the following command:

```bash
twine upload --password="$CIRQ_PYPI_TOKEN" "dist/*"
```

Perform automated verification tests:

```bash
# NOTE: FIRST RUN WILL LIKELY FAIL - pypi might not have yet indexed the version
./dev_tools/packaging/verify-published-package.sh "${VER}" --prod
```

Next, create a Python virtual environment to perform manual verification of the
release:

```bash
mkvirtualenv "verify_${VER}" --python=/usr/bin/python3
pip install cirq
python -c "import cirq; print(cirq.__version__)"
```

### Create the release on GitHub

Using the information above, create the release on the
[Release page](https://github.com/quantumlib/Cirq/releases).
Be sure to include the `.whl` file as an attachment.

### Release PR for notebooks

If there are unreleased notebooks that are under testing (meaning that
`NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES` is not empty in the file
[`dev_tools/notebooks/isolated_notebook_test.py`](dev_tools/notebooks/isolated_notebook_test.py)),
then follow the steps in our [notebooks guide](docs/dev/notebooks.md).

### Verify the Zenodo archive

Each new release should get archived in Zenodo automatically. To check it, [log
in to Zenodo](https://zenodo.org) using credentials stored in Google's internal
password utility (or get someone from Google to do this). Navigate to the [list
of uploads](https://zenodo.org/me/uploads), and ensure an entry for the new
release is present there. Open the page for the entry, verify the information,
and edit it if necessary.

### Email cirq-announce

Lastly, email cirq-announce@googlegroups.com with the release notes and an
announcement of the new version.

Finally, congratulate yourself on a release well done!

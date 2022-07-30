# Versioning and Releases

Below is info on how we version releases, and how the releases
themselves are created. Note that development is done on the `master`
branch, so if you want to use a more stable version you should use one
of the [releases](https://github.com/quantumlib/Cirq/releases) or
install from pypi using `pip install cirq`.  The release from the
latest commit to master can be installed with `pip install --pre cirq`.

## Versioning

We follow [semantic versioning](https://semver.org/) for labeling our
releases.  Versions are labeled MAJOR.MINOR.PATCH where each of these
is a numerical value. The following guarantees are provided:

1. All packages released at the same time from the Cirq repository will share
   the same [Semantic Versioning 2.0.0](https://semver.org/) number.
   1. Packages may have different version numbers in the future, at which point
      this policy will be updated.
2. Libraries in the `cirq-core` directory (with the exception of
   `cirq-core/cirq/contrib`) adhere to the guarantees outlined in the Semantic
   Versioning specification. In summary: “Bug fixes not affecting the API
   increment the patch version, backwards compatible API additions/changes
   increment the minor version, and backwards incompatible API changes increment
   the major version.”
3. The contrib directory (at `cirq-core/cirq/contrib`) follows Semantic
   Versioning except the MINOR version increment policy: releases with MINOR
   version increments may contain backward-incompatible functionality changes to
   its public API.
   1. May be changed to strictly follow Semantic Versioning in the future, at
      which point this policy will be updated.
4. Cirq vendor directories (`cirq-aqt`, `cirq-google`, `cirq-ionq`, etc.) follow
   Semantic Versioning except the MINOR version increment policy: each vendor
   directory has a separate policy on whether MINOR version increments provide
   backward-compatibility guarantees, as described in `version_policy.md` in the
   respective directory.
   1. If `version_policy.md` does not exist in a particular vendor directory,
      MINOR version increments may contain backward-incompatible functionality
      changes to its public API.
   2. For each vendor directory, version policies may be modified to strictly
      follow Semantic Versioning in the future.
5. Versions based on unreleased branches of master will be suffixed with ".dev".

The rules for version changes are:
* Increment the PATCH version if all changes are bug fixes only.
* Increment the MINOR version if changes contain functionalities which are
  backward-compatible, or if a vendor directory or `contrib` contains
  backward-incompatible changes and the policy for the directory allows
  backward-incompatible changes for a minor version increment.

A major version increment process has not been established. Until then,
backward-incompatible changes are not allowed for `cirq-core` and vendor
directories that prohibit them for a minor version increment.

## Releases

We use github's release system for creating releases.  Release are listed
[on the Cirq release page](https://github.com/quantumlib/Cirq/releases).

Our development process uses the `master` branch for development.
Master will always use the next unreleased minor version with the suffix
of ".dev".  When a release is performed, the ".dev" will be removed and tagged
in a release branch with a version tag (vX.X.X).  Then, master will be updated
to the next minor version.  This can always be found in the
[version file](cirq/_version.py).

### Release Schedule
Releases are made on an as-needed basis determined by Cirq maintainers. All Cirq
packages (including vendor packages such as `cirq-aqt`) are released at the same
time.

## Before you release: flush the deprecation backlog

Ensure that all the deprecations are removed that were meant to be deprecated for the given release. 
E.g. if you want to release `v0.11`, you can check with `git grep 'v0.11'` for all the lines containing this deadline.
Make sure none of those are released.  

## Release Procedure

This procedure can be followed by authorized cirq developers to perform a
release.

### Preparation

System requirements: Linux, python3.7

For MINOR / MAJOR release: Make sure you're on an up-to-date master branch and 
in cirq's root directory.

```bash
git checkout master
git pull origin master  # or upstream master
git status  # should be no pending changes
```

For PATCH update: Make sure you checked out the version you want to patch. 
Typically this will be something like `${MAJOR}.${MINOR}.${LAST_PATCH}` 

```bash
git fetch origin # or upstream - to fetch all tags
git checkout <desired tag to patch>   
git status  # should be no pending changes
```

Ensure you have pypi and test pypi accounts with access to cirq distribution.
This can be done by visiting test.pypi.org, logging in, and accessing the cirq
distribution.

For the following script to work, you will need the following env variables
defined: `TEST_TWINE_USERNAME`, `TEST_TWINE_PASSWORD`, `PROD_TWINE_USERNAME`,
`PROD_TWINE_PASSWORD`.

It is highly recommended to use different passwords for test and prod to avoid
accidentally pushing to prod.

Also define these variables for the versions you are releasing:

```bash
VER=VERSION_YOU_WANT_TO_RELEASE  # e.g. "0.7.0"
NEXT_VER=NEXT_VERSION  # e.g. "0.8.0" (skip for PATCH releases)
```

### Create release branch

Create a release branch called "v${VERSION}-dev":

```bash
git checkout -b "v${VER}-dev"
```

If you are doing a PATCH update, also cherrypick the commits for the fixes 
you want to include in your update and resolve all potential merge conflicts 
carefully: 

```bash
git cherry-pick <commit> 
```

Bump the version on the release branch: 

```bash
python dev_tools/modules.py replace_version --old ${VER}.dev --new ${VER} 
git add .
git commit -m "Removing ${VER}.dev -> ${VER}"
git push origin "v${VER}-dev"
```

### Bump the master version 

WARNING: Only bump the master version for minor and major releases, for PATCH
updates, leave it as it is.  

```bash
git checkout master -b "version_bump_${NEXT_VER}"
python dev_tools/modules.py replace_version --old ${VER}.dev --new ${NEXT_VER}.dev
git add .
git commit -m "Bump cirq version to ${NEXT_VER}"
git push origin "version_bump_${NEXT_VER}"
```

Master branch should never see a non-dev version specifier.

### Create distribution wheel

From release branch, create a binary distribution wheel. This is the package
that will go to pypi.

```bash
git checkout "v${VER}-dev"
./dev_tools/packaging/produce-package.sh dist
ls dist  # should only contain one file, for each modules 
```

### Push to test pypi

The package server pypi has a test server where packages can be uploaded to
check that they work correctly before pushing the real version.  This section
illustrates how to upload the package to test pypi and verify that it works.

First, upload the package in the dist/ directory.  (Ensure that this is the only
package in this directory, or modify the commands to upload only this
file).

```bash
twine upload --repository-url=https://test.pypi.org/legacy/ -u="$TEST_TWINE_USERNAME" -p="$TEST_TWINE_PASSWORD" "dist/*"
```

Next, run automated verification.

Note: sometimes the first verification from test pypi will fail.

```bash
# NOTE: FIRST RUN WILL LIKELY FAIL - pypi might not have yet indexed the version
./dev_tools/packaging/verify-published-package.sh "${VER}" --test
```
Once this runs, you can create a virtual environment to perform
manual verification as a sanity check and to check version number and
any high-risk features that have changed this release.

```bash
mkvirtualenv "verify_test_${VER}" --python=/usr/bin/python3
pip install -r dev_tools/requirements/dev.env.txt
pip install --index-url=https://test.pypi.org/simple/ cirq=="${VER}"
python -c "import cirq; print(cirq.__version__)"
python  # just do some stuff checking that latest features are present
```

### Draft release notes and email

Put together a release notes document that can be used as part of a
release and for an announcement email.

You can model the release notes on the previous release from the
[Release page](https://github.com/quantumlib/Cirq/releases).

1. Fill out the new version in "Tag Version" and choose your release 
branch to create the tag from.   
2. Attach the generated whl file to the release 

Retrieve all commits since the last release with:
```git log "--pretty=%h %s"```.

You can get the changes to the top-level objects and protocols by
checking the history of the init files. `git diff <previous version>..HEAD cirq-core/cirq/__init__.py`

You can get the contributing authors for the release by running:
`git log <previous version>..HEAD --pretty="%an" | sort | uniq | sed ':a;N;$!ba;s/\n/, /g'`

### cirq-google Changelog
Add cirq-google release notes to `cirq-google/CHANGELOG.md` following the [changelog format](https://keepachangelog.com/en/1.0.0/)

### Release to prod pypi

Upload to prod pypi using the following command:

```bash
twine upload --username="$PROD_TWINE_USERNAME" --password="$PROD_TWINE_PASSWORD" "dist/*"
```

Perform automated verification tests:

```bash
# NOTE: FIRST RUN WILL LIKELY FAIL - pypi might not have yet indexed the version
./dev_tools/packaging/verify-published-package.sh "${VER}" --prod
```

Next, create a virtual environment to perform manual verification of the
release.

```bash
mkvirtualenv "verify_${VER}" --python=/usr/bin/python3
pip install cirq
python -c "import cirq; print(cirq.__version__)"
```

###  Create the release

Using the information above, create the release on the
[Release page](https://github.com/quantumlib/Cirq/releases).
Be sure to include the whl file as an attachment.

### Release PR for notebooks

If there are unreleased notebooks, that are under testing (`NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES` is not empty in [dev_tools/notebooks/isolated_notebook_test.py](dev_tools/notebooks/isolated_notebook_test.py)), follow the steps in our [notebooks guide](docs/dev/notebooks.md).

### Create zenodo release

Got to the [Zenodo release page](https://zenodo.org/record/6599601#.YpZCspPMLzc).
Login using credentials within Google's internal password utility (or get
someone from Google to do this).  Click "New Version".

*   Upload the new zip file (found in releases page under "assets").
*   Remove old zip file.
*   Update version.
*   Double check all other fields.
*   Click publish.


### Email cirq-announce

Lastly, email cirq-announce@googlegroups.com with the release notes
and an announcement of the new version.

Congratulate yourself for a well done release!

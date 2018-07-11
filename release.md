# Versioning and Releases

Cirq is currently (as of July 11, 2018) alpha, and so has a MAJOR version 
of 0. Below is info on how we version releases, and how the releases 
themselves are created. Note that development is done on the `master` 
branch, so if you want to use a more stable version you should use one 
of the [releases](https://github.com/quantumlib/Cirq/releases) or 
install from pypi.

## Versioning

We follow [semantic versioning](https://semver.org/) for labeling our 
releases.  Versions are labeled MAJOR.MINOR.PATCH where each of these 
is a numerical value. The rules for versions changes are:
* Before MAJOR becomes 1, updates to MINOR can and will make changes to 
public facing apis and interfaces..
* After MAJOR becomes 1, updates that break the public facing api 
or interface need to update  MAJOR version.
* MINOR updates have to be backwards compatible (after MAJOR>=1).
* PATCH updates are for bug fixes.

## Releases

We use github's release system for creating releases.  Release are listed
[on the Cirq release page](https://github.com/quantumlib/Cirq/releases).

Our development process uses the `master` branch for development. 
When a release is made for a major or minor version update, `master`
is tagged with a version tag (vX.X.X) for a pull request corresponding 
to the release.  In the pull request corresponding to the release 
the [version file](cirq/_version.py) should be updated. After
that version is cut, a future pull request should update the 
version to the next minor version with `-dev` appended.

For patch version updates (bug fixes), we follow a different pattern.
For these we create a separate branch that off of the version the
major minor version for the patch, or a previous branch patch.  The
branches  should be of the name `branch-X-X-X` corresponding to the 
version.  These can then be appropriately tagged and the release
pushed to pypi.  These fixes, if possible, should also be merged
into master via a separate change.

         


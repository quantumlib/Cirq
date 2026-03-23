#!/bin/zsh -f

commit=$(git rev-list -1 --abbrev=10 --abbrev-commit --end-of-options ${1:?}) || exit $?


outdir=~/tmp/cov-serial-vs-parallel
idx=$(git rev-list --count ${commit})
outfile=${outdir}/${idx}-${commit}.out
if [[ -s ${outfile} ]]; then
    print -u2 "Bailing out - ${outfile} already exists"
    exit 3
fi

mkdir -p ${outdir}

files=( ${(f)"$(
    git show --name-only --pretty="" --diff-filter=d ${commit} -- '*.py')"}
)

git restore -- "*.py"
( print "#" >>${files} )

check/pytest-changed-files-and-incremental-coverage |& tee ${outfile}

git restore -- "*.py"

#!/bin/bash

# this scripts build the resource tarball when the resources file are presented
# in the directory structure (but untracked by git)

GITROOT=$(git rev-parse --show-toplevel)
if [ -z "$GITROOT" ]; then
    echo "not in a git repository!"
    exit 2
fi


tar -cvzf bliss_models.tar.gz bliss_models/
echo $?

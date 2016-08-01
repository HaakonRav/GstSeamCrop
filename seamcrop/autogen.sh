#!/bin/sh
# you can either set the environment variables AUTOCONF, AUTOHEADER, AUTOMAKE,
# ACLOCAL, AUTOPOINT and/or LIBTOOLIZE to the right versions, or leave them
# unset and get the defaults

if [ -z "$1" ]; then
 echo '\nUsage: Must be run with the compute capability of your GPU: ./autogen.sh <arch>\n';
 echo 'For a GPU with a compute capability of 5.0: ./autogen.sh sm_50';
 echo 'Note: compute capability must be higher than sm_20\n';
 exit 1;
fi

autoreconf --verbose --force --install --make || {
 echo 'autogen.sh failed';
 exit 1;
}

./configure --enable-cuda=$1 || {
 echo 'configure failed';
 exit 1;
}

#cp "$PWD"/moca/.libs/* "$PWD"/src/.libs
#echo "$PWD"/moca/.libs

echo
echo "Now type 'make' to compile this module."
echo

#!/bin/sh

SERVICE="$1"
RESULT=`ps -ef | sed -n /${SERVICE}/p`

if [ "${RESULT:-null}" = null ]; then
    echo "not running"
else
    echo "running"
fi

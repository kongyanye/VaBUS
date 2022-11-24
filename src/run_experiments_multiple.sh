#!/bin/bash

role=$1

if [ $role = "edge" ]; then
    python edge.py param1.yml &
    python edge.py param2.yml &
    python edge.py param3.yml &
elif [ $role = "cloud" ]; then
    python cloud.py param1.yml &
    python cloud.py param2.yml &
    python cloud.py param3.yml &
else
    echo "unknown args"
fi

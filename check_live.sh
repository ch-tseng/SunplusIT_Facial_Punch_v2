#!/bin/bash

getscript() {
  pgrep -lf ".[ /]$1( |\$)"
}

script1=main.py
script2=flask

# test if script 1 is running
if getscript "$script1" >/dev/null; then
  echo "$script1" is RUNNING
  else
    echo "$script1" is NOT running
    sh /home/pi/start.sh > /home/pi/auto_start.log 2>&1
fi

if getscript "$script2" >/dev/null; then
  echo "$script2" is RUNNING
  else
    echo "$script2" is NOT running
    sh /home/pi/works/fdoor.openvino/web/startWeb.sh > /home/pi/auto_start.log 2>&1
fi

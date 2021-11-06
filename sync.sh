#!/bin/bash

rsync -a --exclude=".*" . pi@192.168.1.104:/home/pi/code/roopi

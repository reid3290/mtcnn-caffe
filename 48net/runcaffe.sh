#!/usr/bin/env sh
set -e
~/caffe-master/build/tools/caffe train \
	 --solver=./solver.prototxt \
  	 #--weights=./48net-only-cls.caffemodel

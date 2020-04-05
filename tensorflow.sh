#!/bin/bash
# https://www.tensorflow.org/install/gpu
# https://www.tensorflow.org/install/docker

if [ $# -lt 1 ]
then
	echo "Uso: $0 <version>" 1>&2
	exit
fi

docker run -u $(id -u):$(id -g) --gpus all -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:$1-gpu-py3 bash

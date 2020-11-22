#!/bin/bash

binary=$1
N1=$2
N2=$3
delta=$(((N2-N1)/20))

for i in {1..20}
do
	./${binary} $K $((N1+$i*$delta))
done
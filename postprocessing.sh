#!/bin/bash

clfs=$1

IFS=','
read -rs C <<< $clfs
IFS=' '

for c in ${C[@]}; do
	files=$(find "data_$c/material/" -maxdepth 1 -iname "*.dat")

	IFS=$'\n'
	for i in $files; do
		MetricsComputer.py -i$i -c$c
	done
	IFS=' '

	DataCompactor.py -c$c
	OptimalMetrics.py -c$c
done

IFS=$'\n'
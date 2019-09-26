#!/bin/bash

folder=$1

losses_files=$(find $folder -iname '*losses*')

ModelsLossesGenerator.py $losses_files
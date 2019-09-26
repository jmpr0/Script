#!/bin/bash

folder=$1

summaries=$(find $folder -iname '0_*summary_r*' )
best_summaries=$(find $folder -iname '*summary_best*' | sort)

ComplexityMapGenerator.py $(echo $summaries | wc -w) $summaries $(echo $best_summaries | wc -w) $best_summaries
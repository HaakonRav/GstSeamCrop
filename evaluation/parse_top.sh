#!/bin/bash

indir='raw_data/TOP_'
outdir='parsed_data/top/'

# Parses the output from top into separate files for Memory and CPU load.

for scenario in `seq 2 3`;
do  
  for retarget in `seq 1 3`;
  do
    for size in `seq 1 3`;
    do
      cat $indir$scenario'_'$retarget'_'$size'.txt' | grep evaluationseamc | cut -c 47-52 | nl -i 1 > $outdir'CPU_'$scenario'_'$retarget'_'$size'.txt'
      cat $indir$scenario'_'$retarget'_'$size'.txt' | grep evaluationseamc | cut -c 54-58 | nl -i 1 > $outdir'MEM_'$scenario'_'$retarget'_'$size'.txt'
    done
  done
done
# CPU
#cat $1 | grep gst-launch-1.0 | cut -c 47-52 | nl -i 1 > $1_CPU.txt
# MEM
#cat $1 | grep gst-launch-1.0 | cut -c 54-58 | nl -i 1 > $1_MEM.txt

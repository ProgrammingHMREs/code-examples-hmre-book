#!/usr/bin/env bash 

## Run Square example
#
tornado --jvm="-Dcompute.square.device=0:0" --printKernel --threadInfo -cp target/examples-1.0-SNAPSHOT.jar com.book.hmre.tornadovm.TornadoVMSquare


## Run Reduction
#
tornado --jvm="-Dcompute.reduce.device=0:0" --printKernel --threadInfo -cp target/examples-1.0-SNAPSHOT.jar com.book.hmre.tornadovm.TornadoVMReduction

## Run Matrix Multiplication
#
tornado --jvm="-Dcompute.mxm.device=0:0" --printKernel --threadInfo -cp target/examples-1.0-SNAPSHOT.jar com.book.hmre.tornadovm.TornadoVMMxM


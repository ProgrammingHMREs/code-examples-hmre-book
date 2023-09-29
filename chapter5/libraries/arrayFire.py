#!/usr/bin/env python3
import arrayfire as af
import time 

af.set_backend('cuda')
print(af.info_str())

## Declare a matrix of 1024x1024 
A = af.randu(1024, 1024)

for i in range(1,10):
    start = time.time()
    A2 = af.matmul(A, A)
    af.sync()
    end = time.time()
    print(str(end - start) + " (s)") 

#!/usr/bin/python
import os

# Script for performance testing of GPU

os.system('rm -f performanceData.txt performancePlot.jpeg')

numElems = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

for n in numElems:
	for counter in range(20):
		os.system('./vectorAdd -n ' + str(n) + ' -k 1')

os.system('gnuplot plotter')

all:
	nvcc -c host.cu
	nvcc host.o -o host

run:
	./host 1024

clean:
	rm host *.o

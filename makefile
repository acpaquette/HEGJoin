SOURCES = import_dataset.cpp main.cu SortByWorkload.cu GPU.cu kernel.cu kernel_alt.cu WorkQueue.cpp StaticPartition.cpp
OBJECTS = import_dataset.o WorkQueue.o StaticPartition.o
CUDAOBJECTS = SortByWorkload.o GPU.o kernel.o kernel_alt.o main.o
EGOSOURCES = multiThreadJoin.cpp Util.cpp Point.cpp
EGOBJECTS = Point.o Util.o MultiThreadJoin.o
CC = nvcc
CXX = g++ -g
EXECUTABLE = main

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -arch=compute_86 -code=sm_86 -lcuda -lineinfo -I/home/benoit/research/boost_1_75_0
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
CFLAGS2 = -std=c++14 -O3 -fopenmp -march=native -mavx -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -Wunused -Wlogical-op

all: $(EXECUTABLE)

main.o: main.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) main.cu

SortByWorkload.o: SortByWorkload.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) SortByWorkload.cu

.cpp.o:
	$(CC) $(CFLAGS) $(FLAGS) $(SEARCHMODE) $(PARAMS) $<

WorkQueue.o: WorkQueue.cpp params.h
	$(CXX) $(CFLAGS) $(CFLAGS2) $(SEARCHMODE) $(PARAMS) WorkQueue.cpp

kernel.o: kernel.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) kernel.cu

kernel_alt.o: kernel_alt.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) kernel_alt.cu

GPU.o: GPU.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) GPU.cu

StaticPartition.o: StaticPartition.cpp params.h
	$(CXX) $(CFLAGS) $(CFLAGS2) $(SEARCHMODE) $(PARAMS) StaticPartition.cpp

################################################################################

Point.o: Point.cpp params.h
	$(CXX) -D_REENTRANT $(CFLAGS2) -c Point.cpp

Util.o: Util.cpp params.h
	$(CXX) -D_REENTRANT $(CFLAGS2) -c Util.cpp

MultiThreadJoin.o: MultiThreadJoin.cpp params.h
	$(CXX) -D_REENTRANT $(CFLAGS2) -c MultiThreadJoin.cpp


$(EXECUTABLE): $(CUDAOBJECTS) $(OBJECTS) $(EGOBJECTS)
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm $(EGOBJECTS)
	rm main

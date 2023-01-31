SOURCES = import_dataset.cpp main.cu SortByWorkload.cu GPU.cu kernel.cu WorkQueue.cpp StaticPartition.cpp
OBJECTS = import_dataset.o WorkQueue.o StaticPartition.o
CUDAOBJECTS = SortByWorkload.o GPU.o kernel.o main.o
EGOSOURCES = multiThreadJoin.cpp Util.cpp Point.cpp
EGOBJECTS = Point.o Util.o MultiThreadJoin.o
CC = nvcc
CXX = g++
EXECUTABLE = main

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -arch=compute_80 -code=sm_80 -lpthread -lnvToolsExt -lcuda -lineinfo -I/home/acp263/boost_1_75_0
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
CFLAGS2 = -std=c++14 -O3 -fopenmp -lpthread -march=native -mavx -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -Wunused -Wlogical-op -I/packages/cuda/11.2/include

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

GPU.o: GPU.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) GPU.cu

StaticPartition.o: StaticPartition.cpp params.h
	$(CXX) $(CFLAGS) $(CFLAGS2) $(SEARCHMODE) $(PARAMS) StaticPartition.cpp

################################################################################

Point.o: Point.cpp params.h
	$(CXX) -D_REENTRANT $(CFLAGS2) $(PARAMS) -c Point.cpp

Util.o: Util.cpp params.h
	$(CXX) -D_REENTRANT $(CFLAGS2) $(PARAMS) -c Util.cpp

MultiThreadJoin.o: MultiThreadJoin.cpp params.h
	$(CXX) -D_REENTRANT $(CFLAGS2) $(PARAMS) -c MultiThreadJoin.cpp


$(EXECUTABLE): $(CUDAOBJECTS) $(OBJECTS) $(EGOBJECTS)
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm $(EGOBJECTS)
	rm main

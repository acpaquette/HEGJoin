#ifndef KERNEL_H
#define KERNEL_H

#include "structs.h"
#include "params.h"


__global__ void kernelIndexComputeNonemptyCells(
		DTYPE * database,
		unsigned int * N,
		DTYPE * epsilon,
		DTYPE * minArr,
		unsigned int * nCells,
		uint64_t * pointCellArr,
		unsigned int * databaseVal,
		bool enumerate);


__global__ void sortByWorkLoadGlobal(
		DTYPE * database,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * nNonEmptyCells,
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets,
		schedulingCell * sortedCells);


__global__ void sortByWorkLoadLidUnicomp(
		DTYPE * database,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * nNonEmptyCells,
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets,
		schedulingCell * sortedCells);


__global__ void sortByWorkLoadUnicomp(
		DTYPE * database,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * nNonEmptyCells,
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets,
		schedulingCell * sortedCells);


__device__ uint64_t getLinearID_nDimensionsGPU(
		unsigned int * indexes,
		unsigned int * dimLen,
		unsigned int nDimensions);


__forceinline__ __device__ void evalPoint(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		bool differentCell);


__device__ void evaluateCell(
		unsigned int* nCells,
		unsigned int* indexes,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int* nNonEmptyCells,
		DTYPE* database, DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		DTYPE* point, unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		bool differentCell,
		const unsigned int tpp);


__forceinline__ __device__ void evalPointUnicompOrigin(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx);


__device__ void evaluateCellUnicompOrigin(
		unsigned int* nCells,
		unsigned int* indexes,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int* nNonEmptyCells,
		DTYPE* database, DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		DTYPE* point, unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		unsigned int* nDCellIDs,
		unsigned int nbThreads,
		unsigned int numThread);


__forceinline__ __device__ void evalPointUnicompAdjacent(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx);


__device__ void evaluateCellUnicompAdjacent(
		unsigned int* nCells,
		unsigned int* indexes,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int* nNonEmptyCells,
		DTYPE* database, DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		DTYPE* point, unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		unsigned int* nDCellIDs,
		unsigned int nbThreads,
		unsigned int numThread);


/*############################################################################*/


__global__ void kernelNDGridIndexBatchEstimatorAdaptive(
		unsigned int sampleBegin,
		unsigned int sampleEnd,
		unsigned int *N,
		unsigned int * sampleOffset,
		DTYPE * database,
		DTYPE * sortedDatabase,
		unsigned int * originPointIndex,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells);
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets);


__global__ void kernelNDGridIndexBatchEstimatorAdaptiveTest(
		unsigned int sampleBegin,
		unsigned int sampleEnd,
		unsigned int *N,
		unsigned int * sampleOffset,
		DTYPE * database,
		DTYPE * sortedDatabase,
		unsigned int * originPointIndex,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * estimatedResult);
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets);


__global__ void kernelNDGridIndexBatchEstimatorUnicompAdaptive(
		unsigned int sampleBegin,
		unsigned int sampleEnd,
		unsigned int * N,
		unsigned int * sampleOffset,
		DTYPE * database,
		DTYPE * sortedDatabase,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells);
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets);


__global__ void kernelNDGridIndexBatchEstimatorLidUnicompAdaptive(
		unsigned int sampleBegin,
		unsigned int sampleEnd,
		unsigned int * N,
		unsigned int * sampleOffset,
		DTYPE * database,
		DTYPE * sortedDatabase,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells);
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets);


__global__ void kernelNDGridIndexBatchEstimator_v2(
		unsigned int * N,
		unsigned int * sampleOffset,
		DTYPE * database,
		unsigned int * originPointIndex,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * estimatedResult);


__global__ void kernelNDGridIndexGlobal(
		unsigned int * batchBegin,
		unsigned int * N,
		DTYPE * database,
		DTYPE * sortedCells,
		unsigned int * originPointIndex,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		// unsigned int * gridCellNDMask,
		// unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal,
		const unsigned int tpp);


__global__ void kernelNDGridIndexGlobalUnicomp(
		unsigned int * batchBegin,
		unsigned int * N,
		unsigned int * offset,
		unsigned int * batchNum,
		DTYPE * database,
		DTYPE * sortedCells,
		unsigned int * originPointIndex,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal);


__global__ void kernelNDGridIndexGlobalLinearIDUnicomp(
		unsigned int * batchBegin,
		unsigned int * N,
		unsigned int * offset,
		unsigned int * batchNum,
		DTYPE * database,
		DTYPE * sortedCells,
		unsigned int * originPointIndex,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE * minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal);


#endif

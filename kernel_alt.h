#ifndef KERNEL_ALT_H
#define KERNEL_ALT_H

#include <cuda_fp16.h>

#include "structs.h"
#include "params.h"

__device__ uint64_t getLinearID_nDimensionsGPUKernelAlt(
	unsigned int* indexes,
	unsigned int* dimLen,
	unsigned int nDimensions);

__global__ void convertAndResizeDataset(
    float* in,
    half* out,
    unsigned int nbQueries);

__global__ void convertMinArr(
	DTYPE* in,
	half* out);

__global__ void convertFloatToHalf2(
    float* input,
    half2* tmp,
    half2* output,
    unsigned int nbPoints);

__device__ void evaluateCell_alt(
	unsigned int* nCells,
	unsigned int* indexes,
	struct gridCellLookup* gridCellLookupArr,
	unsigned int* nNonEmptyCells,
	half* database,
	DTYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	half* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx,
	unsigned int* nDCellIDs);

__forceinline__ __device__ void evalPoint_alt(
	unsigned int* gridLookupArr,
	int k,
	half* database,
	DTYPE* epsilon,
	half* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx);

__forceinline__ __device__ void evalPointILP_alt(
	unsigned int* gridLookupArr,
	int k,
	half* database,
	DTYPE* epsilon,
	half* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx);

__global__ void batchEstimatorKernel_alt(
	unsigned int* N,
	unsigned int* sampleOffset,
	DTYPE* database,
	unsigned int* originPointIndex,
	DTYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	DTYPE* minArr,
	unsigned int* nCells,
	unsigned int* cnt,
	unsigned int* nNonEmptyCells,
	unsigned int* estimatedResult,
	unsigned int* candidatesCounter);

__global__ void distanceCalculationBruteForceCuda(
    DTYPE* database,
	unsigned int* nbQueries,
    unsigned int* queryOffset,
    DTYPE* epsilon,
    unsigned int* nbNeighbors);

__global__ void distanceCalculationBruteForceCudaHalf(
	half* database,
	unsigned int* nbQueries,
	unsigned int* queryOffset,
	DTYPE* epsilon,
	unsigned int* nbNeighbors);

__global__ void distanceCalculationBruteForceCuda_half2(
	half2* database,
	unsigned int* nbQueries,
	unsigned int* queryOffset,
	DTYPE* epsilon,
	unsigned int* nbNeighbors);

__global__ void distanceCalculationBruteForceTensor_TwoStepsComputePagingOneQuery(
	half* dataset,
	unsigned int* nbQueries,
	half* identity,
	DTYPE* epsilon,
	unsigned int* nbNeighbors);

__global__ void distanceCalculationBruteForceTensor_TwoStepsComputePagingOneQueryOptim(
	half* dataset,
	unsigned int* nbQueries,
	half* identity,
	DTYPE* epsilon,
	unsigned int* nbNeighbors);

__global__ void distanceCalculationBruteForceTensor_OneStepComputePagingOneQuery(
	half* dataset,
	unsigned int* nbQueries,
	DTYPE* epsilon,
	unsigned int* nbNeighbors);

__global__ void distanceCalculationBruteForceTensor_OneStepComputePagingOneQueryOptim(
	half* dataset,
	unsigned int* nbQueries,
	DTYPE* epsilon,
	unsigned int* nbNeighbors);

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

__global__ void distanceCalculationGridCudaHalf(
    unsigned int* batchBegin,
    unsigned int* batchSize,
    half* database,
    unsigned int* originPointIndex,
    DTYPE* epsilon,
    struct grid* grid,
    unsigned int* gridLookupArr,
    struct gridCellLookup* gridCellLookupArr,
    half* minArr,
    unsigned int* nCells,
    unsigned int* cnt,
    unsigned int* nNonEmptyCells,
    int* pointIDKey,
    int* pointInDistVal);

__global__ void distanceCalculationGridCudaHalf2(
    unsigned int* batchBegin,
    unsigned int* batchSize,
    half2* database,
    unsigned int* originPointIndex,
    DTYPE* epsilon,
    struct grid* grid,
    unsigned int* gridLookupArr,
    struct gridCellLookup* gridCellLookupArr,
    half* minArr,
    unsigned int* nCells,
    unsigned int* cnt,
    unsigned int* nNonEmptyCells,
    int* pointIDKey,
    int* pointInDistVal);

// __global__ void distanceCalculationGridTensor(
//     unsigned int* batchBegin,
//     unsigned int* batchSize,
//     half* database,
//     unsigned int* originPointIndex,
//     DTYPE* epsilon,
//     struct grid* grid,
//     unsigned int* gridLookupArr,
//     struct gridCellLookup* gridCellLookupArr,
//     half* minArr,
//     unsigned int* nCells,
//     unsigned int* cnt,
//     unsigned int* nNonEmptyCells,
//     int* pointIDKey,
//     int* pointInDistVal,
//     half* identityMatrix);

__global__ void distanceCalculationGridTensor_TwoStepsComputePagingOneQuery(
	unsigned int* batchBegin,
	unsigned int* batchSize,
	half* database,
	unsigned int* originPointIndex,
	half* identityMatrix,
	DTYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	half* minArr,
	unsigned int* nCells,
	unsigned int* cnt,
	unsigned int* nNonEmptyCells,
	int* pointIDKey,
	int* pointInDistVal);

__global__ void distanceCalculationGridTensor_OneStepComputePagingOneQuery(
	unsigned int* batchBegin,
	unsigned int* batchSize,
	half* database,
	unsigned int* originPointIndex,
	DTYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	half* minArr,
	unsigned int* nCells,
	unsigned int* cnt,
	unsigned int* nNonEmptyCells,
	int* pointIDKey,
	int* pointInDistVal);

#endif

#include <math.h>
#include <stdio.h>

#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#include "kernel_alt.h"
#include "params.h"
#include "structs.h"

// Specific to tensor cores
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
using namespace nvcuda;
using namespace cooperative_groups;



__device__ uint64_t getLinearID_nDimensionsGPUKernelAlt(
	unsigned int* indexes,
	unsigned int* dimLen,
	unsigned int nDimensions)
{
    uint64_t offset = 0;
	uint64_t multiplier = 1;

	for (int i = 0; i < nDimensions; ++i)
	{
		offset += (uint64_t) indexes[i] * multiplier;
		multiplier *= dimLen[i];
	}

	return offset;
}


__global__ void convertAndResizeDataset(
    DTYPE* in,
    half* out,
    unsigned int nbQueries)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nbQueries)
    {
	    // Copy the coordinates from the dataset
        for (int i = 0; i < GPUNUMDIM; ++i)
        {
            out[tid * COMPUTE_DIM + i] = (half)in[tid * GPUNUMDIM + i];
        }
		// Fill with 0s so the dimensionality of the dataset is a multiple of 16
		for (int i = GPUNUMDIM; i < COMPUTE_DIM; ++i)
		{
			out[tid * COMPUTE_DIM + i] = (half)0.0;
		}
    }
    // The original dataset does not have the 15 supplemental points, so need to do it in another step
    if (tid < 15)
    {
	    // Create "fake points" with 0s coordinates so the last query point will still have 15 points after when loading using load_matrix_sync
		for (int i = 0; i < COMPUTE_DIM; ++i)
		{
			out[tid * COMPUTE_DIM + i] = (half)0.0;
		}
    }
}



__global__ void convertMinArr(
	DTYPE* in,
	half* out)
{
	for (int i = 0; i < NUMINDEXEDDIM; ++i)
	{
		out[i] = (half)in[i];
	}
}


__global__ void convertMinArrHalf2(
	DTYPE* in,
	half2* out)
{
	for (int i = 0; i < NUMINDEXEDDIM / 2; ++i)
	{
		out[i] = __floats2half2_rn(in[i * 2], in[i * 2 + 1]);
	}
}


__global__ void convertFloatToHalf2(
    float* input,
    // half2* tmp,
    half2* output,
    unsigned int nbPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nbPoints)
    {
		if (0 == (GPUNUMDIM % 2))
		{
			for (int i = 0; i < HALF2_DIM; ++i)
			{
				output[tid * HALF2_DIM + i] = __floats2half2_rn(input[tid * GPUNUMDIM + i * 2], input[tid * GPUNUMDIM + i * 2 + 1]);
			}

			// for (int i = 0; i < GPUNUMDIM; i += 2)
			// {
			// 	output[tid * HALF2_DIM + (i / 2)] = __floats2half2_rn(input[tid * GPUNUMDIM + i], input[tid * GPUNUMDIM + i + 1]);
			// }
		} else {
			for (int i = 0; i < (GPUNUMDIM / 2); ++i)
			{
				output[tid * HALF2_DIM + i] = __floats2half2_rn(input[tid * GPUNUMDIM + i * 2], input[tid * GPUNUMDIM + i * 2 + 1]);
			}
			output[tid * HALF2_DIM + (GPUNUMDIM / 2) + 1] = __floats2half2_rn(input[tid * GPUNUMDIM + GPUNUMDIM - 1], 0.0f);
		}
    }
}


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
	unsigned int* candidatesCounter)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ((*N) <= tid)
	{
		return;
	}

	unsigned int pointId = tid * (*sampleOffset);

	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; ++i)
	{
		point[i] = database[ originPointIndex[pointId] * GPUNUMDIM + i ];
	}

	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; ++i)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1);
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1);
	}

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	unsigned int localNeighborCounter = 0;
	unsigned int localCandidateCounter = 0;

	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; ++x)
			{
				indexes[x] = loopRng[x];
			}

			uint64_t calcLinearID = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, NUMINDEXEDDIM);
			//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
			//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

			struct gridCellLookup tmp;
			tmp.gridLinearID = calcLinearID;

			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;

				for (int k = grid[GridIndex].indexmin; k <= grid[GridIndex].indexmax; ++k)
				{
					DTYPE runningTotalDist = 0;
					unsigned int dataIdx = gridLookupArr[k];

					for (int l = 0; l < GPUNUMDIM; ++l)
					{
						runningTotalDist += (database[dataIdx * GPUNUMDIM + l]  - point[l])
								* (database[dataIdx * GPUNUMDIM + l] - point[l]);
					}

					#if ACCUM_PREC == 16
					if (hsqrt(runningTotalDist) <= (*epsilon))
					#else
					if (sqrt(runningTotalDist) <= (*epsilon))
					#endif
					{
						unsigned int idx = atomicAdd(cnt, int(1));
						localNeighborCounter++;
					}
				}

				localCandidateCounter += grid[GridIndex].indexmax - grid[GridIndex].indexmin + 1;
			}
		} //end loop body

	estimatedResult[tid] = localNeighborCounter;
	candidatesCounter[tid] = localCandidateCounter;
}


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
	unsigned int* nDCellIDs)
{
	// Compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	// a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
                gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		for(int k = grid[GridIndex].indexmin; k <= grid[GridIndex].indexmax; ++k)
        {
			#if ILP == 1
				evalPoint_alt(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			#else
				evalPointILP_alt(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			#endif
		}
	}
}


__device__ void evaluateCellHalf2(
	unsigned int* nCells,
	unsigned int* indexes,
	struct gridCellLookup* gridCellLookupArr,
	unsigned int* nNonEmptyCells,
	half2* database,
	DTYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	half2* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx,
	unsigned int* nDCellIDs)
{
	// Compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	// a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
                gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		for(int k = grid[GridIndex].indexmin; k <= grid[GridIndex].indexmax; ++k)
        {
			#if ILP == 1
				evalPointHalf2(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			#else
				evalPointHalf2ILP(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			#endif
		}
	}
}


__forceinline__ __device__ void evalPoint_alt(
	unsigned int* gridLookupArr,
	int k,
	half* database,
	DTYPE* epsilon,
	half* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	DTYPE runningTotalDist = 0;
	unsigned int dataIdx = gridLookupArr[k];

	for(int l = 0; l < GPUNUMDIM; ++l)
    {
		runningTotalDist += (DTYPE)((database[dataIdx * COMPUTE_DIM + l] - point[l]) * (database[dataIdx * COMPUTE_DIM + l] - point[l]));
	}

    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
    #if DTYPE_PREC == 16
	if(hsqrt(runningTotalDist) <= (*epsilon))
    #else
    if(sqrt(runningTotalDist) <= (*epsilon))
    #endif
    {
		unsigned int idx = atomicAdd(cnt, int(1));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
	}
}


__forceinline__ __device__ void evalPointILP_alt(
	unsigned int* gridLookupArr,
	int k,
	half* database,
	DTYPE* epsilon,
	half* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	unsigned int dataIdx = gridLookupArr[k];
	DTYPE runningTotalDist[ILP];

	// const unsigned int unrollSize = ILP;

	#pragma unroll
	for (int i = 0; i < ILP; ++i)
	{
		runningTotalDist[i] = 0.0;
	}

	for(int i = 0; i < GPUNUMDIM; i += ILP)
    {
		#pragma unroll
		for (int j = 0; j < ILP && (i + j) < GPUNUMDIM; ++j)
		{
			runningTotalDist[j] += (DTYPE)((database[dataIdx * COMPUTE_DIM + i + j] - point[i + j]) * (database[dataIdx * COMPUTE_DIM + i + j] - point[i + j]));
		}

		#if SHORT_CIRCUIT
			#pragma unroll
			for (int j = 1; j < ILP; ++j)
			{
				runningTotalDist[0] += runningTotalDist[j];
				runningTotalDist[j] = 0.0;
			}

			#if DTYPE_PREC == 16
			if (hsqrt(runningTotalDist[0]) > (*epsilon))
			#else
			if (sqrt(runningTotalDist[0]) > (*epsilon))
			#endif
			{
				return;
			}
		#endif
	}

	#if !SHORT_CIRCUIT
		#pragma unroll
		for (int i = 1; i < ILP; ++i)
		{
			runningTotalDist[0] += runningTotalDist[i];
		}
	#endif

    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
    #if DTYPE_PREC == 16
	if(hsqrt(runningTotalDist[0]) <= (*epsilon))
    #else
    if(sqrt(runningTotalDist[0]) <= (*epsilon))
    #endif
    {
		unsigned int idx = atomicAdd(cnt, int(1));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
	}
}


__forceinline__ __device__ void evalPointHalf2(
	unsigned int* gridLookupArr,
	int k,
	half2* database,
	DTYPE* epsilon,
	half2* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	unsigned int dataIdx = gridLookupArr[k];

	#if DTYPE_PREC == 16
		half runningTotalDist = 0.0;
		for (int i = 0; i < HALF2_DIM; ++i)
		{
			half2 tmpDistance = __hsub2(__hmul2(point[i], database[dataIdx * HALF2_DIM + i]),
										__hmul2(point[i], database[dataIdx * HALF2_DIM + i]));
			runningTotalDist += __low2half(tmpDistance) + __high2half(tmpDistance);
		}
	#else
		float runningTotalDist = 0.0;
		for (int i = 0; i < HALF2_DIM; ++i)
		{
			float2 tmpDistance = __half22float2(__hsub2(__hmul2(point[i], database[dataIdx * HALF2_DIM + i]),
														__hmul2(point[i], database[dataIdx * HALF2_DIM + i])));
			runningTotalDist += tmpDistance.x + tmpDistance.y;
		}
	#endif

    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
    #if DTYPE_PREC == 16
	if(hsqrt(runningTotalDist) <= (*epsilon))
    #else
    if(sqrt(runningTotalDist) <= (*epsilon))
    #endif
    {
		unsigned int idx = atomicAdd(cnt, int(1));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
	}
}


__forceinline__ __device__ void evalPointHalf2ILP(
	unsigned int* gridLookupArr,
	int k,
	half2* database,
	DTYPE* epsilon,
	half2* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	#if ILP > 1

	unsigned int dataIdx = gridLookupArr[k];
	DTYPE runningTotalDist[ILP / 2];

	// const unsigned int unrollSize = ILP;

	#pragma unroll
	for (int i = 0; i < ILP / 2; ++i)
	{
		runningTotalDist[i] = (DTYPE)0.0;
	}

	for(int i = 0; i < HALF2_DIM; i += (ILP / 2))
    {
		#pragma unroll
		for (int j = 0; j < (ILP / 2) && (i + j) < HALF2_DIM; ++j)
		{
			#if DTYPE_PREC == 16
				half2 tmpDistance = __hsub2(__hmul2(point[i + j], database[dataIdx * HALF2_DIM + i + j]),
											__hmul2(point[i + j], database[dataIdx * HALF2_DIM + i + j]));
				runningTotalDist[j] += __low2half(tmpDistance) + __high2half(tmpDistance);
			#else
				float2 tmpDistance = __half22float2(__hsub2(__hmul2(point[i + j], database[dataIdx * HALF2_DIM + i + j]),
															__hmul2(point[i + j], database[dataIdx * HALF2_DIM + i + j])));
				runningTotalDist[j] += tmpDistance.x + tmpDistance.y;
			#endif
		}

		#if SHORT_CIRCUIT
			#pragma unroll
			for (int j = 1; j < (ILP / 2); ++j)
			{
				runningTotalDist[0] += runningTotalDist[j];
				runningTotalDist[j] = (DTYPE)0.0;
			}

			#if DTYPE_PREC == 16
			if (hsqrt(runningTotalDist[0]) > (*epsilon))
			#else
			if (sqrt(runningTotalDist[0]) > (*epsilon))
			#endif
			{
				return;
			}
		#endif
	}

	#if !SHORT_CIRCUIT
		#pragma unroll
		for (int i = 1; i < (ILP / 2); ++i)
		{
			runningTotalDist[0] += runningTotalDist[i];
		}
	#endif

    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
    #if DTYPE_PREC == 16
	if(hsqrt(runningTotalDist[0]) <= (*epsilon))
    #else
    if(sqrt(runningTotalDist[0]) <= (*epsilon))
    #endif
    {
		unsigned int idx = atomicAdd(cnt, int(1));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
	}

	#endif
}



//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\



// __global__ void distanceCalculationBruteForceCuda(
//     DTYPE* database,
// 	unsigned int* nbQueries,
//     unsigned int* queryOffset,
//     DTYPE* epsilon,
//     unsigned int* nbNeighbors)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if ((*nbQueries) <= tid)
//     {
//         return;
//     }
//
//     DTYPE point[GPUNUMDIM];
//     for (int i = 0; i < GPUNUMDIM; ++i)
//     {
//         point[i] = database[tid * COMPUTE_DIM + i];
//     }
//
//     DTYPE runningTotalDist[ILP];
// 	const unsigned int unrollSize = ILP;
//
// 	for (int i = 0; i < (*nbQueries); ++i)
// 	{
// 		#pragma unroll unrollSize
// 		for (int j = 0; j < ILP; ++j)
// 		{
// 			runningTotalDist[j] = 0.0;
// 		}
//
// 		for(int j = 0; j < GPUNUMDIM; j += ILP)
// 	    {
// 			#pragma unroll unrollSize
// 			for (int k = 0; k < ILP && (j + k) < GPUNUMDIM; ++k)
// 			{
// 				runningTotalDist[k] += (database[i * COMPUTE_DIM + j + k] - point[j + k]) * (database[i * COMPUTE_DIM + j + k] - point[j + k]);
// 			}
//
// 			#if SHORT_CIRCUIT
// 				#pragma unroll (unrollSize - 1)
// 				for (int k = 1; k < ILP; ++k)
// 				{
// 					runningTotalDist[0] += runningTotalDist[k];
// 					runningTotalDist[k] = 0.0;
// 				}
//
// 				#if ACCUM_PREC == 16
// 				if (hsqrt(runningTotalDist[0]) > (*epsilon))
// 				#else
// 				if (sqrt(runningTotalDist[0]) > (*epsilon))
// 				#endif
// 				{
// 					return;
// 				}
// 			#endif
// 		}
//
// 		#if !SHORT_CIRCUIT
// 			#pragma unroll unrollSize
// 			for (int j = 1; j < ILP; ++j)
// 			{
// 				runningTotalDist[0] += runningTotalDist[j];
// 			}
// 		#endif
//
// 	    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
// 	    #if DTYPE_PREC == 16
// 		if(hsqrt(runningTotalDist[0]) <= (*epsilon))
// 	    #else
// 	    if(sqrt(runningTotalDist[0]) <= (*epsilon))
// 	    #endif
// 	    {
// 			atomicAdd(nbNeighbors, int(1));
// 		}
// 	}
// }


__global__ void distanceCalculationBruteForceCudaHalf(
	half* database,
	unsigned int* nbQueries,
	unsigned int* queryOffset,
	DTYPE* epsilon,
	unsigned int* nbNeighbors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((*nbQueries) <= tid)
	{
		return;
	}

	half point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; ++i)
	{
		point[i] = database[tid * COMPUTE_DIM + i];
	}

	for (int i = 0; i < (*nbQueries); ++i)
	{
		DTYPE resultDistance = 0.0;

		for (int j = 0; j < GPUNUMDIM; ++j)
		{
			resultDistance += (DTYPE)((point[j] * database[i * COMPUTE_DIM + j]) - (point[j] * database[i * COMPUTE_DIM + j]));
		}

		#if DTYPE_PREC == 16
		if (hsqrt(resultDistance) <= (*epsilon))
		#else
		if (sqrt(resultDistance) <= (*epsilon))
		#endif
		{
			atomicAdd(nbNeighbors, int(1));
		}
	}
}



__global__ void distanceCalculationBruteForceCuda_half2(
	half2* database,
	unsigned int* nbQueries,
	unsigned int* queryOffset,
	DTYPE* epsilon,
	unsigned int* nbNeighbors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((*nbQueries) <= tid)
	{
		return;
	}

	half2 point[HALF2_DIM];
	for (int i = 0; i < HALF2_DIM; ++i)
	{
		point[i] = database[tid * HALF2_DIM + i];
	}

	for (int i = 0; i < (*nbQueries); ++i)
	{
		#if DTYPE_PREC == 16
			half resultDistance = 0.0;
			for (int j = 0; j < HALF2_DIM; ++j)
			{
				half2 tmpResult = __hsub2(__hmul2(point[j], database[i * HALF2_DIM + i]), __hmul2(point[j], database[i * HALF2_DIM + i]));
				resultDistance += __low2half(tmpResult) + __high2half(tmpResult);
			}
		#else
			float resultDistance = 0.0;
			for (int j = 0; j < HALF2_DIM; ++j)
			{
				half2 tmpResult = __hsub2(__hmul2(point[j], database[i * HALF2_DIM + i]), __hmul2(point[j], database[i * HALF2_DIM + i]));
				resultDistance += __low2float(tmpResult) + __high2float(tmpResult);
			}
		#endif

		#if DTYPE_PREC == 16
		if (hsqrt(resultDistance) <= (*epsilon))
		#else
		if (sqrt(resultDistance) <= (*epsilon))
		#endif
		{
			atomicAdd(nbNeighbors, int(1));
		}
	}
}



__global__ void distanceCalculationBruteForceTensor_TwoStepsComputePagingOneQuery(
	half* dataset,
	unsigned int* nbQueries,
	half* identity,
	DTYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ DTYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> identityFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half> firstStepAccumulator;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, DTYPE> secondStepAccumulator;

	wmma::load_matrix_sync(identityFragment, identity, TILE_SIZE_HALF);

	for (int i = 0; i < POINTS_PER_WARP; ++i)
	{
		unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbDimsToPage; ++j)
		{
			if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
			{
				sharedArrayQueryPoint[warpIdInBlock * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j] =
					dataset[(tid / WARP_SIZE) * POINTS_PER_WARP * COMPUTE_DIM + i * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j];
			}
		}

		for (int j = 0; j < (*nbQueries); j += TILE_SIZE_HALF)
		{
			wmma::fill_fragment(secondStepAccumulator, 0.0);

			for (int k = 0; k < COMPUTE_DIM; k += TILE_SIZE_HALF)
			{
				wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoint + (warpIdInBlock * COMPUTE_DIM + k), 0);

				wmma::load_matrix_sync(firstStepAccumulator, dataset + j * COMPUTE_DIM + k, COMPUTE_DIM, wmma::mem_row_major);
				for (int l = 0; l < firstStepAccumulator.num_elements; ++l)
				{
					firstStepAccumulator.x[l] = (half)-1.0 * firstStepAccumulator.x[l];
				}

				wmma::mma_sync(firstStepAccumulator, matrixAFragment, identityFragment, firstStepAccumulator);

				wmma::store_matrix_sync(sharedArrayResultFirstStep + sharedArrayResultOffset, firstStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);
				wmma::load_matrix_sync(matrixAFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);
				wmma::load_matrix_sync(matrixBFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);

				wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
			}

			wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

			if (warp.thread_rank() < TILE_SIZE_HALF)
			{
				DTYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if DTYPE_PREC == 16
				if(hsqrt(resultDistance) <= (*epsilon))
				#else
				if(sqrt(resultDistance) <= (*epsilon))
				#endif
				{
					atomicAdd(nbNeighbors, int(1));
				}
			}
			warp.sync();
		}
	}
}



__global__ void distanceCalculationBruteForceTensor_TwoStepsComputePagingOneQueryOptim(
	half* dataset,
	unsigned int* nbQueries,
	half* identity,
	DTYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ DTYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> identityFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half> firstStepAccumulator;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, DTYPE> secondStepAccumulator;

	wmma::load_matrix_sync(identityFragment, identity, TILE_SIZE_HALF);

	for (int i = 0; i < POINTS_PER_WARP; ++i)
	{
		unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbStepsToPage; ++j)
		{
			if ((j * WARP_SIZE + warp.thread_rank()) < COMPUTE_DIM)
			{
				sharedArrayQueryPoint[warpIdInBlock * COMPUTE_DIM + j * WARP_SIZE + warp.thread_rank()] =
					dataset[(tid / WARP_SIZE) * POINTS_PER_WARP * COMPUTE_DIM + i * COMPUTE_DIM + j * WARP_SIZE + warp.thread_rank()];
			}
		}

		for (int j = 0; j < (*nbQueries); j += TILE_SIZE_HALF)
		{
			wmma::fill_fragment(secondStepAccumulator, 0.0);

			for (int k = 0; k < COMPUTE_DIM; k += TILE_SIZE_HALF)
			{
				wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoint + (warpIdInBlock * COMPUTE_DIM + k), 0);

				wmma::load_matrix_sync(firstStepAccumulator, dataset + j * COMPUTE_DIM + k, COMPUTE_DIM, wmma::mem_row_major);
				for (int l = 0; l < firstStepAccumulator.num_elements; ++l)
				{
					firstStepAccumulator.x[l] = (half)-1.0 * firstStepAccumulator.x[l];
				}

				wmma::mma_sync(firstStepAccumulator, matrixAFragment, identityFragment, firstStepAccumulator);

				wmma::store_matrix_sync(sharedArrayResultFirstStep + sharedArrayResultOffset, firstStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);
				wmma::load_matrix_sync(matrixAFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);
				wmma::load_matrix_sync(matrixBFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);

				wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
			}

			wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

			if (warp.thread_rank() < TILE_SIZE_HALF)
			{
				DTYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if DTYPE_PREC == 16
				if(hsqrt(resultDistance) <= (*epsilon))
				#else
				if(sqrt(resultDistance) <= (*epsilon))
				#endif
				{
					atomicAdd(nbNeighbors, int(1));
				}
			}
			warp.sync();
		}
	}
}



__global__ void distanceCalculationBruteForceTensor_OneStepComputePagingOneQuery(
	half* dataset,
	unsigned int* nbQueries,
	DTYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayTmpFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ DTYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, DTYPE> secondStepAccumulator;

	for (int i = 0; i < POINTS_PER_WARP; ++i)
	{
		unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbDimsToPage; ++j)
		{
			if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
			{
				sharedArrayQueryPoint[warpIdInBlock * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j] =
					dataset[(tid / WARP_SIZE) * POINTS_PER_WARP * COMPUTE_DIM + i * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j];
			}
		}

		for (int j = 0; j < (*nbQueries); j += TILE_SIZE_HALF)
		{
			wmma::fill_fragment(secondStepAccumulator, 0.0);

			for (int k = 0; k < COMPUTE_DIM; k += TILE_SIZE_HALF)
			{
				unsigned int nbElemsToPage = (TILE_SIZE_HALF * TILE_SIZE_HALF) / WARP_SIZE;
				unsigned int threadPerPoint = WARP_SIZE / TILE_SIZE_HALF;
				for (int l = 0; l < nbElemsToPage; ++l)
				{
					sharedArrayTmpFirstStep[sharedArrayResultOffset + warp.thread_rank() * nbElemsToPage + l] =
						sharedArrayQueryPoint[warpIdInBlock * COMPUTE_DIM + k + (warp.thread_rank() / threadPerPoint) * nbElemsToPage + l]
						- dataset[j * COMPUTE_DIM + k + (warp.thread_rank() / threadPerPoint) * COMPUTE_DIM + (warp.thread_rank() / threadPerPoint) * nbElemsToPage + l];
				}

				wmma::load_matrix_sync(matrixAFragment, sharedArrayTmpFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);
				wmma::load_matrix_sync(matrixBFragment, sharedArrayTmpFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);

				wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
			}

			wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

			if (warp.thread_rank() < TILE_SIZE_HALF)
			{
				DTYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if DTYPE_PREC == 16
				if(hsqrt(resultDistance) <= (*epsilon))
				#else
				if(sqrt(resultDistance) <= (*epsilon))
				#endif
				{
					atomicAdd(nbNeighbors, int(1));
				}
			}
			warp.sync();
		}
	}
}



__global__ void distanceCalculationBruteForceTensor_OneStepComputePagingOneQueryOptim(
	half* dataset,
	unsigned int* nbQueries,
	DTYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayTmpFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ DTYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, DTYPE> secondStepAccumulator;

	for (int i = 0; i < POINTS_PER_WARP; ++i)
	{
		unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbStepsToPage; ++j)
		{
			if ((j * WARP_SIZE + warp.thread_rank()) < COMPUTE_DIM)
			{
				sharedArrayQueryPoint[warpIdInBlock * COMPUTE_DIM + j * WARP_SIZE + warp.thread_rank()] =
					dataset[(tid / WARP_SIZE) * POINTS_PER_WARP * COMPUTE_DIM + i * COMPUTE_DIM + j * WARP_SIZE + warp.thread_rank()];
			}
		}

		for (int j = 0; j < (*nbQueries); j += TILE_SIZE_HALF)
		{
			wmma::fill_fragment(secondStepAccumulator, 0.0);

			for (int k = 0; k < COMPUTE_DIM; k += TILE_SIZE_HALF)
			{
				unsigned int nbStepsOfPage = (TILE_SIZE_HALF * TILE_SIZE_HALF) / WARP_SIZE;
				for (int l = 0; l < nbStepsOfPage; ++l)
				{
					sharedArrayTmpFirstStep[sharedArrayResultOffset + l * WARP_SIZE + warp.thread_rank()] =
						sharedArrayQueryPoint[warpIdInBlock * COMPUTE_DIM + k + (warp.thread_rank() % TILE_SIZE_HALF)]
						- dataset[j * COMPUTE_DIM + k + 2 * l * COMPUTE_DIM + (warp.thread_rank() / TILE_SIZE_HALF) * COMPUTE_DIM + warp.thread_rank()];
				}

				wmma::load_matrix_sync(matrixAFragment, sharedArrayTmpFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);
				wmma::load_matrix_sync(matrixBFragment, sharedArrayTmpFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);

				wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
			}

			wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

			if (warp.thread_rank() < TILE_SIZE_HALF)
			{
				DTYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if DTYPE_PREC == 16
				if(hsqrt(resultDistance) <= (*epsilon))
				#else
				if(sqrt(resultDistance) <= (*epsilon))
				#endif
				{
					atomicAdd(nbNeighbors, int(1));
				}
			}
			warp.sync();
		}
	}
}



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
    int* pointInDistVal)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*batchSize) <= tid)
    {
        return;
    }

    // Get the next query point in the "local" queue
    unsigned int pointId = atomicAdd(batchBegin, int(1));

    half point[GPUNUMDIM];
    for (int i = 0; i < GPUNUMDIM; ++i)
    {
        point[i] = database[ originPointIndex[pointId] * COMPUTE_DIM + i ];
    }

    // Calculate the coords of the Cell for the point and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
    unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

    for (int i = 0; i < NUMINDEXEDDIM; ++i)
    {
        nDCellIDs[i] = (DTYPE)(point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); // Boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); // Boundary conditions (don't go beyond the maximum number of cells)
    }

    unsigned int indexes[NUMINDEXEDDIM];
    unsigned int loopRng[NUMINDEXEDDIM];

    for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++)
			{
				indexes[x] = loopRng[x];
			}

			evaluateCell_alt(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, grid,
					gridLookupArr, point, cnt, pointIDKey, pointInDistVal, originPointIndex[pointId], nDCellIDs);

		} //end loop body
}



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
    int* pointInDistVal)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*batchSize) <= tid)
    {
        return;
    }

    // Get the next query point in the "local" queue
    unsigned int pointId = atomicAdd(batchBegin, int(1));

    half2 point[HALF2_DIM];
    for (int i = 0; i < HALF2_DIM; ++i)
    {
        point[i] = database[ originPointIndex[pointId] * HALF2_DIM + i ];
    }

    // Calculate the coords of the Cell for the point and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
    unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

    for (int i = 0; i < NUMINDEXEDDIM; ++i)
    {
		if (0 == (i % 2))
		{
			nDCellIDs[i] = (DTYPE)(__low2half(point[i / 2]) - minArr[i]) / (*epsilon);
		} else {
			nDCellIDs[i] = (DTYPE)(__high2half(point[i / 2]) - minArr[i]) / (*epsilon);
		}
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); // Boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); // Boundary conditions (don't go beyond the maximum number of cells)
    }

    unsigned int indexes[NUMINDEXEDDIM];
    unsigned int loopRng[NUMINDEXEDDIM];

    for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++)
			{
				indexes[x] = loopRng[x];
			}

			evaluateCellHalf2(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, grid,
					gridLookupArr, point, cnt, pointIDKey, pointInDistVal, originPointIndex[pointId], nDCellIDs);

		} //end loop body
}



__global__ void distanceCalculationGridTensor_TwoStepsComputePagingOneQuery(
	unsigned int* batchBegin,
	unsigned int* batchEnd,
	half* database,
	unsigned int* nbQueryPoints,
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
	int* pointInDistVal)
{
	__shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ DTYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> identityFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half> firstStepAccumulator;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, DTYPE> secondStepAccumulator;

	wmma::load_matrix_sync(identityFragment, identityMatrix, TILE_SIZE_HALF);

	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	unsigned int firstQueryId;
	if (0 == warp.thread_rank())
	{
		firstQueryId = atomicAdd(batchBegin, int(POINTS_PER_WARP));
	}
	firstQueryId = __shfl_sync(0xffffffff, firstQueryId, 0);

	for (int i = firstQueryId; i < (firstQueryId + POINTS_PER_WARP) && i < (*batchEnd); ++i)
	{
		unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbStepsToPage; ++j)
		{
			if ((j * WARP_SIZE + warp.thread_rank()) < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j * WARP_SIZE + warp.thread_rank()] =
					database[originPointIndex[i] * COMPUTE_DIM + j * WARP_SIZE + warp.thread_rank()];
			}
		}

		for (int j = 0; j < NUMINDEXEDDIM; ++j)
		{
			nDCellIDs[j] = (sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j] - minArr[j]) / (half)(*epsilon);
			nDMinCellIDs[j] = max(0, nDCellIDs[j] - 1);
			nDMaxCellIDs[j] = min(nCells[j] - 1, nDCellIDs[j] + 1);
		}

		DTYPE resultDistance = 0.0;

		for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
			for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
			#include "kernelloops.h"
			{ //beginning of loop body
				for (int x = 0; x < NUMINDEXEDDIM; ++x)
				{
					indexes[x] = loopRng[x];
				}

				uint64_t cellLinearId = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, NUMINDEXEDDIM);
				struct gridCellLookup tmp;
				tmp.gridLinearID = cellLinearId;

				// Find if the neighboring cell is empty or not
				if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
				{
					struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
						gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
					unsigned int gridIndex = resultBinSearch->idx;

					// For all the candidates in the cell, process them 16 by 16
					for (int k = grid[gridIndex].indexmin; k <= grid[gridIndex].indexmax; k += TILE_SIZE_HALF)
					{
						unsigned int nbCandidatesLeft = grid[gridIndex].indexmax - k + 1;

						wmma::fill_fragment(secondStepAccumulator, 0.0f);

						// For all dimensions of the candidates, process them 16 by 16
						for (int n = 0; n < COMPUTE_DIM; n += TILE_SIZE_HALF)
						{
							wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoints + warpIdInBlock * COMPUTE_DIM + n, 0);

							// unsigned int candidateId;
							// thread_block_tile<16> halfWarp = tiled_partition<16>(warp);
							unsigned int halfWarpId = warp.thread_rank() / 16;
							unsigned int halfWarpThreadId = warp.thread_rank() % 16;

							for (int j = 0; j < TILE_SIZE_HALF; j += 2)
							{
								unsigned int candidateId;
								// if ((k + j + halfWarp.meta_group_rank()) < (*nbQueryPoints))
								// {
								// 	candidateId = gridLookupArr[k + j + halfWarp.meta_group_rank()];
								// } else {
								// 	candidateId = 0;
								// }
								if ((k + j + halfWarpId) < (*nbQueryPoints))
								{
									candidateId = gridLookupArr[k + j + halfWarpId];
								} else {
									candidateId = 0;
								}

								sharedArrayResultFirstStep[sharedArrayResultOffset + (j + halfWarpId) * TILE_SIZE_HALF + halfWarpThreadId] =
									database[candidateId * COMPUTE_DIM + n + halfWarpThreadId];
							}

							wmma::load_matrix_sync(firstStepAccumulator, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF, wmma::mem_row_major);
							for (int j = 0; j < firstStepAccumulator.num_elements; ++j)
							{
								firstStepAccumulator.x[j] = (half)-1.0 * firstStepAccumulator.x[j];
							}

							wmma::mma_sync(firstStepAccumulator, matrixAFragment, identityFragment, firstStepAccumulator);
							wmma::store_matrix_sync(sharedArrayResultFirstStep + sharedArrayResultOffset, firstStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

							wmma::load_matrix_sync(matrixAFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);
							wmma::load_matrix_sync(matrixBFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);

							wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);

							#if SHORT_CIRCUIT
								wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

								int nbThreadsShortCircuit = 0;
								if (warp.thread_rank() < TILE_SIZE_HALF && warp.thread_rank() < nbCandidatesLeft)
								{
									resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

									int shortCircuit = 0;
									#if DTYPE_PREC == 16
									if (hsqrt(resultDistance) > (*epsilon))
									#else
									if (sqrt(resultDistance) > (*epsilon))
									#endif
									{
										shortCircuit = 1;
									}

									// Match if all 16 candidate points short-circuited
									__match_all_sync(__activemask(), shortCircuit, &nbThreadsShortCircuit);
								}

								// Get from thread 0 if the threads that computed the distances short-circuited
								nbThreadsShortCircuit = __shfl_sync(0xffffffff, nbThreadsShortCircuit, 0);
								if (nbThreadsShortCircuit)
								{
									// Break the loop iterating over the dimensions of the current candidates
									n = COMPUTE_DIM;
								}
							#endif
						} // for COMPUTE_DIM

						#if SHORT_CIRCUIT
							if (warp.thread_rank() < TILE_SIZE_HALF && warp.thread_rank() < nbCandidatesLeft)
							{
								// The distance was already computed on the last short-circuit check
								#if DTYPE_PREC == 16
								if (hsqrt(resultDistance) <= (*epsilon))
								#else
								if (sqrt(resultDistance) <= (*epsilon))
								#endif
								{
									unsigned int tmpIdx = atomicAdd(cnt, int(1));
									pointIDKey[tmpIdx] = originPointIndex[i];
									pointInDistVal[tmpIdx] = gridLookupArr[k];
								}
							}
						#else
							wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

							if (warp.thread_rank() < TILE_SIZE_HALF && warp.thread_rank() < nbCandidatesLeft)
							{
								// resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];
								resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank()];

								#if DTYPE_PREC == 16
								if(hsqrt(resultDistance) <= (*epsilon))
								#else
								if(sqrt(resultDistance) <= (*epsilon))
								#endif
								{
									unsigned int tmpIdx = atomicAdd(cnt, int(1));
									pointIDKey[tmpIdx] = originPointIndex[i];
									pointInDistVal[tmpIdx] = gridLookupArr[k];
								}
							}
							warp.sync();
						#endif
					} // for candidates
				} // if non-empty cell
			} // for neighboring cells
	} // for query points
}



// __global__ void distanceCalculationGridTensor_OneStepComputePagingOneQuery(
// 	unsigned int* batchBegin,
// 	unsigned int* batchSize,
// 	half* database,
// 	unsigned int* originPointIndex,
// 	DTYPE* epsilon,
// 	struct grid* grid,
// 	unsigned int* gridLookupArr,
// 	struct gridCellLookup* gridCellLookupArr,
// 	half* minArr,
// 	unsigned int* nCells,
// 	unsigned int* cnt,
// 	unsigned int* nNonEmptyCells,
// 	int* pointIDKey,
// 	int* pointInDistVal)
// {
// 	__shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * COMPUTE_DIM];
// 	__shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
// 	__shared__ DTYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
//
// 	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
// 	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
// 	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//
// 	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
//
// 	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
// 	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
// 	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, DTYPE> secondStepAccumulator;
//
// 	unsigned int nDCellIDs[NUMINDEXEDDIM];
// 	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
// 	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
// 	unsigned int indexes[NUMINDEXEDDIM];
// 	unsigned int loopRng[NUMINDEXEDDIM];
//
// 	unsigned int firstQueryId;
// 	if (0 == warp.thread_rank())
// 	{
// 		firstQueryId = atomicAdd(batchBegin, int(POINTS_PER_WARP));
// 	}
// 	firstQueryId = __shfl_sync(0xffffffff, firstQueryId, 0);
//
// 	for (int i = firstQueryId; i < (firstQueryId + POINTS_PER_WARP); ++i)
// 	{
// 		unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
// 		for (int j = 0; j < nbDimsToPage; ++j)
// 		{
// 			if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
// 			{
// 				sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j] =
// 					database[ originPointIndex[i] * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j];
// 			}
// 		}
//
// 		for (int j = 0; j < NUMINDEXEDDIM; ++j)
// 		{
// 			nDCellIDs[j] = (sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j] - minArr[j]) / (half)(*epsilon);
// 			nDMinCellIDs[j] = max(0, nDCellIDs[j] - 1);
// 			nDMaxCellIDs[j] = min(nCells[j] - 1, nDCellIDs[j] + 1);
// 		}
//
// 		DTYPE resultDistance = 0.0;
//
// 		for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
// 			for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
// 			#include "kernelloops.h"
// 			{ //beginning of loop body
// 				for (int x = 0; x < NUMINDEXEDDIM; ++x)
// 				{
// 					indexes[x] = loopRng[x];
// 				}
//
// 				uint64_t cellLinearId = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, NUMINDEXEDDIM);
// 				struct gridCellLookup tmp;
// 				tmp.gridLinearID = cellLinearId;
//
// 				// Find if the neighboring cell is empty or not
// 				if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
// 				{
// 					struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
// 						gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
// 					unsigned int gridIndex = resultBinSearch->idx;
//
// 					for (int k = grid[gridIndex].indexmin; k <= grid[gridIndex].indexmax; k += TILE_SIZE_HALF)
// 					{
// 						unsigned int nbCandidatesLeft = grid[gridIndex].indexmax - k + 1;
//
// 						wmma::fill_fragment(secondStepAccumulator, 0.0f);
//
// 						unsigned int candidateId = gridLookupArr[k + (warp.thread_rank() / 2)];
//
// 						for (int n = 0; n < COMPUTE_DIM; n += TILE_SIZE_HALF)
// 						{
// 							wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoints + (warpIdInBlock * COMPUTE_DIM + k), 0);
//
// 							// unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
// 							for (int j = 0; j < nbDimsToPage; ++j)
// 							{
// 								if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
// 								{
// 									sharedArrayResultFirstStep[sharedArrayResultOffset + warp.thread_rank() * nbDimsToPage + j] =
// 										database[candidateId * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j] * (half)-1.0;
// 								}
// 							}
// 							wmma::load_matrix_sync(matrixBFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);
//
// 							wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
// 						}
//
// 						wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);
//
// 						if (warp.thread_rank() < TILE_SIZE_HALF)
// 						{
// 							resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];
//
// 							#if DTYPE_PREC == 16
// 							if(hsqrt(resultDistance) <= (*epsilon))
// 							#else
// 							if(sqrt(resultDistance) <= (*epsilon))
// 							#endif
// 							{
// 								unsigned int tmpIdx = atomicAdd(cnt, int(1));
// 								pointIDKey[tmpIdx] = originPointIndex[i];
// 								pointInDistVal[tmpIdx] = candidateId;
// 							}
// 						}
// 						warp.sync();
// 					}
// 				}
// 			}
// 	}
// }

#include <math.h>
#include <stdio.h>

#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#include "gpu_join_kernels.cuh"
#include "utils.cuh"
#include "params.h"
#include "structs.h"

// Specific to tensor cores
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
using namespace nvcuda;
using namespace cooperative_groups;



__device__ uint64_t getLinearID_nDimensionsGPUKernel(
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


__global__ void convertDataset(
    float* in,
    half* out,
    unsigned int nbQueries)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nbQueries)
    {
        int pointId = tid * COMPUTE_DIM;
        for (int i = 0; i < COMPUTE_DIM; ++i)
        {
            out[pointId + i] = (half)in[pointId + i];
        }
    }
}


__global__ void convertDataset1D(
    float* in,
    half* out,
    unsigned int nbQueries)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nbQueries)
    {
        out[tid] = (half)in[tid];
    }
}


__global__ void convertFloatToHalf2(
    float* input,
    half2* tmp,
    half2* output,
    unsigned int nbPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nbPoints)
    {
        tmp[tid] = __float2half2_rn(input[tid]);
    }
    if (tid < (nbPoints / 2))
    {
        output[tid] = __lows2half2(tmp[tid * 2], tmp[tid * 2 + 1]);
    }
}


__device__ void evaluateCell(
	unsigned int* nCells,
	unsigned int* indexes,
	struct gridCellLookup* gridCellLookupArr,
	unsigned int* nNonEmptyCells,
	ACCUM_TYPE* database,
	ACCUM_TYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx,
	unsigned int* nDCellIDs)
{
	// Compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	// a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPUKernel(indexes, nCells, INDEXED_DIM);

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
				evalPoint(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			#else
				evalPointILP(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			#endif
		}
	}
}


__forceinline__ __device__ void evalPoint(
	unsigned int* gridLookupArr,
	int k,
	ACCUM_TYPE* database,
	ACCUM_TYPE* epsilon,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	ACCUM_TYPE runningTotalDist = 0;
	unsigned int dataIdx = gridLookupArr[k];

	for(int l = 0; l < INPUT_DATA_DIM; ++l)
    {
		runningTotalDist += (ACCUM_TYPE)(((COMPUTE_TYPE)database[dataIdx * COMPUTE_DIM + l] - point[l])
				* ((COMPUTE_TYPE)database[dataIdx * COMPUTE_DIM + l] - point[l]));
	}

    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
    #if ACCUM_PREC == 16
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


__forceinline__ __device__ void evalPointILP(
	unsigned int* gridLookupArr,
	int k,
	ACCUM_TYPE* database,
	ACCUM_TYPE* epsilon,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	unsigned int dataIdx = gridLookupArr[k];
	ACCUM_TYPE runningTotalDist[ILP];

	const unsigned int unrollSize = ILP;

	#pragma unroll unrollSize
	for (int i = 0; i < ILP; ++i)
	{
		runningTotalDist[i] = 0.0;
	}

	for(int i = 0; i < INPUT_DATA_DIM; i += ILP)
    {
		#pragma unroll unrollSize
		for (int j = 0; j < ILP && (i + j) < INPUT_DATA_DIM; ++j)
		{
			runningTotalDist[j] += (ACCUM_TYPE)(((COMPUTE_TYPE)database[dataIdx * COMPUTE_DIM + i + j] - point[i + j])
				* ((COMPUTE_TYPE)database[dataIdx * COMPUTE_DIM + i + j] - point[i + j]));
		}

		#if SHORT_CIRCUIT
			#pragma unroll (unrollSize - 1)
			for (int j = 1; j < ILP; ++j)
			{
				runningTotalDist[0] += runningTotalDist[j];
				runningTotalDist[j] = 0.0;
			}

			#if ACCUM_PREC == 16
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
		#pragma unroll unrollSize
		for (int i = 1; i < ILP; ++i)
		{
			runningTotalDist[0] += runningTotalDist[i];
		}
	#endif

    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
    #if ACCUM_PREC == 16
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



//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\



__global__ void batchEstimatorKernel(
	unsigned int* N,
	unsigned int* sampleOffset,
	ACCUM_TYPE* database,
	unsigned int* originPointIndex,
	ACCUM_TYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	ACCUM_TYPE* minArr,
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

	COMPUTE_TYPE point[INPUT_DATA_DIM];
	for (int i = 0; i < INPUT_DATA_DIM; ++i)
	{
		point[i] = (COMPUTE_TYPE)database[ originPointIndex[pointId] * COMPUTE_DIM + i ];
	}

	unsigned int nDCellIDs[INDEXED_DIM];
	unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];

	for (int i = 0; i < INDEXED_DIM; ++i)
	{
		nDCellIDs[i] = ((ACCUM_TYPE)point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1);
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1);
	}

	unsigned int indexes[INDEXED_DIM];
	unsigned int loopRng[INDEXED_DIM];

	unsigned int localNeighborCounter = 0;
	unsigned int localCandidateCounter = 0;

	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < INDEXED_DIM; ++x)
			{
				indexes[x] = loopRng[x];
			}

			uint64_t calcLinearID = getLinearID_nDimensionsGPUKernel(indexes, nCells, INDEXED_DIM);
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
					ACCUM_TYPE runningTotalDist = 0;
					unsigned int dataIdx = gridLookupArr[k];

					for (int l = 0; l < INPUT_DATA_DIM; ++l)
					{
						runningTotalDist += (ACCUM_TYPE)(((COMPUTE_TYPE)database[dataIdx * COMPUTE_DIM + l]  - point[l])
								* ((COMPUTE_TYPE)database[dataIdx * COMPUTE_DIM + l] - point[l]));
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



//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\



__global__ void distanceCalculationBruteForceCuda(
    ACCUM_TYPE* database,
	unsigned int* nbQueries,
    unsigned int* queryOffset,
    ACCUM_TYPE* epsilon,
    unsigned int* nbNeighbors)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*nbQueries) <= tid)
    {
        return;
    }

    COMPUTE_TYPE point[INPUT_DATA_DIM];
    for (int i = 0; i < INPUT_DATA_DIM; ++i)
    {
        point[i] = database[tid * COMPUTE_DIM + i];
    }

    ACCUM_TYPE runningTotalDist[ILP];
	const unsigned int unrollSize = ILP;

	for (int i = 0; i < (*nbQueries); ++i)
	{
		#pragma unroll unrollSize
		for (int j = 0; j < ILP; ++j)
		{
			runningTotalDist[j] = 0.0;
		}

		for(int j = 0; j < INPUT_DATA_DIM; j += ILP)
	    {
			#pragma unroll unrollSize
			for (int k = 0; k < ILP && (j + k) < INPUT_DATA_DIM; ++k)
			{
				runningTotalDist[k] += (ACCUM_TYPE)(((COMPUTE_TYPE)database[i * COMPUTE_DIM + j + k] - point[j + k])
					* ((COMPUTE_TYPE)database[i * COMPUTE_DIM + j + k] - point[j + k]));
			}

			#if SHORT_CIRCUIT
				#pragma unroll (unrollSize - 1)
				for (int k = 1; k < ILP; ++k)
				{
					runningTotalDist[0] += runningTotalDist[k];
					runningTotalDist[k] = 0.0;
				}

				#if ACCUM_PREC == 16
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
			#pragma unroll unrollSize
			for (int j = 1; j < ILP; ++j)
			{
				runningTotalDist[0] += runningTotalDist[j];
			}
		#endif

	    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
	    #if ACCUM_PREC == 16
		if(hsqrt(runningTotalDist[0]) <= (*epsilon))
	    #else
	    if(sqrt(runningTotalDist[0]) <= (*epsilon))
	    #endif
	    {
			atomicAdd(nbNeighbors, int(1));
		}
	}
}


// __global__ void distanceCalculationBruteForceTensor_tmp(
//     ACCUM_TYPE* database,
// 	unsigned int* datasetSize,
// 	unsigned int* queryOffset,
//     ACCUM_TYPE* epsilon, // Either half or float
//     unsigned int* nbNeighbors)
// {
// 	__shared__ half sharedArrayQueryPoints[COMPUTE_DIM * WARP_PER_BLOCK];
// 	__shared__ half sharedArrayFirstStep[TILE_SIZE * TILE_SIZE * WARP_PER_BLOCK];
// 	__shared__ ACCUM_TYPE sharedArrayResSecondStep[TILE_SIZE * TILE_SIZE * WARP_PER_BLOCK];
//
// 	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int warpId = threadIdx.x / WARP_SIZE;
//
// 	unsigned int sharedMemoryOffset = warpId * TILE_SIZE * TILE_SIZE;
//
// 	wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> pointsFragment;
// 	wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> candidatesFragment;
// 	wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, ACCUM_TYPE> distanceAccumulatorFragment;
//
// 	for (int i = 0; i < POINTS_PER_WARP; ++i)
// 	{
// 		// Thread 0 of the warp gets the next point to compute, and shares it with the rest of the warp
// 		unsigned int pointId;
// 		if (0 == warp.thread_rank())
// 		{
// 			pointId = atomicAdd(queryOffset, int(1));
// 		}
// 		pointId = __shfl_sync(0xffffffff, pointId, 0);
//
// 		// Copy the query point into shared memory
// 		unsigned int nbDimToCopy = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
// 		for (int j = 0; j < nbDimToCopy; ++j)
// 		{
// 			if (warp.thread_rank() * nbDimToCopy < COMPUTE_DIM)
// 			{
// 				sharedArrayQueryPoints[warpId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j] =
// 					(half)database[ pointId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j ];
// 				// sharedArrayQueryPoints[warpId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j] =
// 				// 	database[ pointId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j ];
// 			}
// 		}
// 		warp.sync();
//
// 		// wmma::load_matrix_sync(identityFragment, identityMatrix, TILE_SIZE);
//
// 		ACCUM_TYPE resultDistance = 0.0;
// 		for (int j = 0; j < (*datasetSize); j += TILE_SIZE)
// 		{
// 			unsigned int nbCandidatesLeft = (*datasetSize) - j;
//
// 			wmma::fill_fragment(distanceAccumulatorFragment, 0.0f);
//
// 			for (int k = 0; k < COMPUTE_DIM / TILE_SIZE; k += TILE_SIZE)
// 			{
// 				unsigned int dimOffset = (TILE_SIZE * TILE_SIZE) * WARP_SIZE;
// 				for (int l = 0; l < dimOffset; ++l)
// 				{
// 					// Page into shared memory the difference between the query point and the candidate points coordinates
// 					sharedArrayFirstStep[sharedMemoryOffset + warp.thread_rank() * dimOffset + l]
// 						= sharedArrayQueryPoints[warpId * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l];
// 						- (half)database[j * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l];
// 					// sharedArrayFirstStep[sharedMemoryOffset + warp.thread_rank() * dimOffset + l]
// 					// 	= sharedArrayQueryPoints[warpId * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l]
// 					// 	- database[j * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l];
// 				}
// 				warp.sync();
//
// 				wmma::load_matrix_sync(pointsFragment, sharedArrayFirstStep + sharedMemoryOffset, TILE_SIZE);
// 				wmma::load_matrix_sync(candidatesFragment, sharedArrayFirstStep + sharedMemoryOffset, TILE_SIZE);
//
// 				wmma::mma_sync(distanceAccumulatorFragment, pointsFragment, candidatesFragment, distanceAccumulatorFragment);
// 				#if SHORT_CIRCUIT
// 					// Store the distance between the query point and the candidate points into shared memory
// 					wmma::store_matrix_sync(sharedArrayResSecondStep + sharedMemoryOffset, distanceAccumulatorFragment, TILE_SIZE, wmma::mem_row_major);
//
// 					// Extract the distance between the query point and the candidate points
// 					// These 16 distances are located in the diagonal of the matrix
// 					int threadsShortCircuit = 0;
// 					if (warp.thread_rank() < TILE_SIZE && warp.thread_rank() < nbCandidatesLeft)
// 					{
// 						resultDistance = sharedArrayResSecondStep[sharedMemoryOffset + warp.thread_rank() * TILE_SIZE + warp.thread_rank()];
//
// 						// Check if this thread should short-circuit
// 						int shortCircuit = 0;
// 						#if ACCUM_PREC == 16
// 						if (hsqrt(resultDistance) > (*epsilon))
// 						#else
// 						if (sqrt(resultDistance) > (*epsilon))
// 						#endif
// 						{
// 							shortCircuit = 1;
// 						}
//
// 						// int nbThreadsShortCircuit = 0;
// 						// Match if all 16 candidate points short-circuited
// 						__match_all_sync(__activemask(), shortCircuit, &threadsShortCircuit);
// 					}
//
// 					// Get from thread 0 if the threads that computed the distances short-circuited
// 					threadsShortCircuit = __shfl_sync(0xffffffff, threadsShortCircuit, 0);
// 					if (threadsShortCircuit)
// 					{
// 						k = COMPUTE_DIM;
// 					}
// 				#endif
// 			} // For steps over the dimensions
//
// 			#if SHORT_CIRCUIT
// 				if (warp.thread_rank() < TILE_SIZE && warp.thread_rank() < nbCandidatesLeft)
// 				{
// 					// The distance was already computed on the last short-circuit check
// 					#if ACCUM_PREC == 16
// 					if (hsqrt(resultDistance) <= (*epsilon))
// 					#else
// 					if (sqrt(resultDistance) <= (*epsilon))
// 					#endif
// 					{
// 						atomicAdd(nbNeighbors, int(1));
// 					}
// 				}
// 			#else
// 				// Store the distance between the query point and the candidate points into shared memory
// 				wmma::store_matrix_sync(sharedArrayResSecondStep + sharedMemoryOffset, distanceAccumulatorFragment, TILE_SIZE, wmma::mem_row_major);
//
// 				// Extract the distance between the query point and the candidate points
// 				// These 16 distances are located in the diagonal of the matrix
// 				if (warp.thread_rank() < TILE_SIZE && warp.thread_rank() < nbCandidatesLeft)
// 				{
// 					resultDistance = sharedArrayResSecondStep[sharedMemoryOffset + warp.thread_rank() * TILE_SIZE + warp.thread_rank()];
//
// 					// Discriminate the two cases, as a different function should be used depending on the precision
// 					#if ACCUM_PREC == 16
// 					if (hsqrt(resultDistance) <= (*epsilon))
// 					#else
// 					if (sqrt(resultDistance) <= (*epsilon))
// 					#endif
// 					{
// 						atomicAdd(nbNeighbors, int(1));
// 					}
// 				}
// 			#endif
//
// 			nbCandidatesLeft -= TILE_SIZE;
// 		} // for all points in dataset
// 	} // for query points of the warp
// }
//
//
// __global__ void distanceCalculationBruteForceTensor(
//     half* database,
// 	unsigned int* datasetSize,
// 	unsigned int* queryOffset,
//     ACCUM_TYPE* epsilon, // Either half or float
//     unsigned int* nbNeighbors)
// {
// 	__shared__ half sharedArrayQueryPoints[COMPUTE_DIM * WARP_PER_BLOCK];
// 	__shared__ half sharedArrayFirstStep[TILE_SIZE * TILE_SIZE * WARP_PER_BLOCK];
// 	__shared__ ACCUM_TYPE sharedArrayResSecondStep[TILE_SIZE * TILE_SIZE * WARP_PER_BLOCK];
//
// 	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int warpId = threadIdx.x / WARP_SIZE;
//
// 	unsigned int sharedMemoryOffset = warpId * TILE_SIZE * TILE_SIZE;
//
// 	wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> pointsFragment;
// 	wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> candidatesFragment;
// 	wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, ACCUM_TYPE> distanceAccumulatorFragment;
//
// 	for (int i = 0; i < POINTS_PER_WARP; ++i)
// 	{
// 		// Thread 0 of the warp gets the next point to compute, and shares it with the rest of the warp
// 		unsigned int pointId;
// 		if (0 == warp.thread_rank())
// 		{
// 			pointId = atomicAdd(queryOffset, int(1));
// 		}
// 		pointId = __shfl_sync(0xffffffff, pointId, 0);
//
// 		// Copy the query point into shared memory
// 		unsigned int nbDimToCopy = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
// 		for (int j = 0; j < nbDimToCopy; ++j)
// 		{
// 			if (warp.thread_rank() * nbDimToCopy < COMPUTE_DIM)
// 			{
// 				sharedArrayQueryPoints[warpId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j] =
// 					(half)database[ pointId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j ];
// 				// sharedArrayQueryPoints[warpId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j] =
// 				// 	database[ pointId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j ];
// 			}
// 		}
// 		warp.sync();
//
// 		// wmma::load_matrix_sync(identityFragment, identityMatrix, TILE_SIZE);
//
// 		ACCUM_TYPE resultDistance = 0.0;
// 		for (int j = 0; j < (*datasetSize); j += TILE_SIZE)
// 		{
// 			unsigned int nbCandidatesLeft = (*datasetSize) - j;
//
// 			wmma::fill_fragment(distanceAccumulatorFragment, 0.0f);
//
// 			for (int k = 0; k < COMPUTE_DIM / TILE_SIZE; k += TILE_SIZE)
// 			{
// 				unsigned int dimOffset = (TILE_SIZE * TILE_SIZE) * WARP_SIZE;
// 				for (int l = 0; l < dimOffset; ++l)
// 				{
// 					// Page into shared memory the difference between the query point and the candidate points coordinates
// 					sharedArrayFirstStep[sharedMemoryOffset + warp.thread_rank() * dimOffset + l]
// 						= sharedArrayQueryPoints[warpId * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l];
// 						// - (half)database[j * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l];
// 					// sharedArrayFirstStep[sharedMemoryOffset + warp.thread_rank() * dimOffset + l]
// 					// 	= sharedArrayQueryPoints[warpId * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l]
// 					// 	- database[j * COMPUTE_DIM + k * TILE_SIZE + (warp.thread_rank() % 2) * dimOffset + l];
// 				}
// 				warp.sync();
//
// 				wmma::load_matrix_sync(pointsFragment, sharedArrayFirstStep + sharedMemoryOffset, TILE_SIZE);
// 				wmma::load_matrix_sync(candidatesFragment, sharedArrayFirstStep + sharedMemoryOffset, TILE_SIZE);
//
// 				wmma::mma_sync(distanceAccumulatorFragment, pointsFragment, candidatesFragment, distanceAccumulatorFragment);
// 				#if SHORT_CIRCUIT
// 					// Store the distance between the query point and the candidate points into shared memory
// 					wmma::store_matrix_sync(sharedArrayResSecondStep + sharedMemoryOffset, distanceAccumulatorFragment, TILE_SIZE, wmma::mem_row_major);
//
// 					// Extract the distance between the query point and the candidate points
// 					// These 16 distances are located in the diagonal of the matrix
// 					int threadsShortCircuit = 0;
// 					if (warp.thread_rank() < TILE_SIZE && warp.thread_rank() < nbCandidatesLeft)
// 					{
// 						resultDistance = sharedArrayResSecondStep[sharedMemoryOffset + warp.thread_rank() * TILE_SIZE + warp.thread_rank()];
//
// 						// Check if this thread should short-circuit
// 						int shortCircuit = 0;
// 						#if ACCUM_PREC == 16
// 						if (hsqrt(resultDistance) > (*epsilon))
// 						#else
// 						if (sqrt(resultDistance) > (*epsilon))
// 						#endif
// 						{
// 							shortCircuit = 1;
// 						}
//
// 						// int nbThreadsShortCircuit = 0;
// 						// Match if all 16 candidate points short-circuited
// 						__match_all_sync(__activemask(), shortCircuit, &threadsShortCircuit);
// 					}
//
// 					// Get from thread 0 if the threads that computed the distances short-circuited
// 					threadsShortCircuit = __shfl_sync(0xffffffff, threadsShortCircuit, 0);
// 					if (threadsShortCircuit)
// 					{
// 						k = COMPUTE_DIM;
// 					}
// 				#endif
// 			} // For steps over the dimensions
//
// 			#if SHORT_CIRCUIT
// 				if (warp.thread_rank() < TILE_SIZE && warp.thread_rank() < nbCandidatesLeft)
// 				{
// 					// The distance was already computed on the last short-circuit check
// 					#if ACCUM_PREC == 16
// 					if (hsqrt(resultDistance) <= (*epsilon))
// 					#else
// 					if (sqrt(resultDistance) <= (*epsilon))
// 					#endif
// 					{
// 						atomicAdd(nbNeighbors, int(1));
// 					}
// 				}
// 			#else
// 				// Store the distance between the query point and the candidate points into shared memory
// 				wmma::store_matrix_sync(sharedArrayResSecondStep + sharedMemoryOffset, distanceAccumulatorFragment, TILE_SIZE, wmma::mem_row_major);
//
// 				// Extract the distance between the query point and the candidate points
// 				// These 16 distances are located in the diagonal of the matrix
// 				if (warp.thread_rank() < TILE_SIZE && warp.thread_rank() < nbCandidatesLeft)
// 				{
// 					resultDistance = sharedArrayResSecondStep[sharedMemoryOffset + warp.thread_rank() * TILE_SIZE + warp.thread_rank()];
//
// 					// Discriminate the two cases, as a different function should be used depending on the precision
// 					#if ACCUM_PREC == 16
// 					if (hsqrt(resultDistance) <= (*epsilon))
// 					#else
// 					if (sqrt(resultDistance) <= (*epsilon))
// 					#endif
// 					{
// 						atomicAdd(nbNeighbors, int(1));
// 					}
// 				}
// 			#endif
//
// 			nbCandidatesLeft -= TILE_SIZE;
// 		} // for all points in dataset
// 	} // for query points of the warp
// }
//
//
//
// __global__ void distanceCalculationBruteForceTensor_v1(
//     half* dataset,
//     unsigned int* nbQueries,
//     half* identity,
//     ACCUM_TYPE* epsilon, // Either half or float
//     unsigned int* nbNeighbors)
// {
// 	// Shared memory array to page the query points
//     __shared__ half sharedQueryPoints[TILE_SIZE * COMPUTE_DIM * WARP_PER_BLOCK];
//     // Each warp needs a full tile in shared memory for temporary results
//     __shared__ half sharedTmpArrayHalf[TILE_SIZE * TILE_SIZE * WARP_PER_BLOCK]; // Used to store the result of the first step
//     __shared__ ACCUM_TYPE sharedTmpArray[TILE_SIZE * TILE_SIZE * WARP_PER_BLOCK]; // Used to store the result of the second step
//
//     int sharedMemoryTmpOffset = (threadIdx.x / WARP_SIZE) * TILE_SIZE * TILE_SIZE;
//     int sharedMemoryPointOffset = (threadIdx.x / WARP_SIZE) * TILE_SIZE * COMPUTE_DIM;
//
//     thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int warpId = tid / WARP_SIZE;
//
//     wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> point_frag;
//     wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> i_frag;
//     wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, half> points_acc;
//     // Accumulate the last matrix multiplication (the actual distances) in single precision
//     wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, ACCUM_TYPE> tmp_dist_acc;
//
//     // Load the identity matrix I
//     wmma::load_matrix_sync(i_frag, identity, TILE_SIZE);
//
//     // Page the query points into shared memory
//     // Each warp has 16 points, so a thread page half the dimensionality of a point
//     // As DATADIM is supposed to be a multiple of TILE_SIZE (16 here), which is a power of 2, DATADIM should
//     //   always be divisible by 2...
//     int warpOffset = COMPUTE_DIM / 2;
//     for (int i = 0; i < warpOffset; ++i)
//     {
//         sharedQueryPoints[sharedMemoryPointOffset + warp.thread_rank() * warpOffset + i] =
//             dataset[warpId * POINTS_PER_WARP * COMPUTE_DIM + warp.thread_rank() * warpOffset + i];
//     }
//     warp.sync();
//
//     // For all the query points assigned to the warp
//     for (int i = 0; i < POINTS_PER_WARP; ++i)
//     {
//         // For all the points
//         for (int j = 0; j < (*nbQueries); j += TILE_SIZE)
//         {
//             // Set the result matrix to 0
//             wmma::fill_fragment(tmp_dist_acc, 0.0f);
//
//             // Tile the computation on the data dimensionality
//             // MEM_DIM should ALWAYS be a multiple of TILE_SIZE
//             for (int n = 0; n < COMPUTE_DIM / TILE_SIZE; ++n)
//             {
//                 // Load 16 dimensions of the query point i
//                 wmma::load_matrix_sync(point_frag, sharedQueryPoints + sharedMemoryPointOffset + i * COMPUTE_DIM + n * TILE_SIZE, 0);
//
//                 // Load all the points, and scale them by -1 to substract, instead of adding (when performing A*B+C)
//                 wmma::load_matrix_sync(points_acc, dataset + j * COMPUTE_DIM + n * TILE_SIZE, TILE_SIZE, wmma::mem_row_major);
//                 for (int k = 0; k < points_acc.num_elements; ++k)
//                 {
//                     points_acc.x[k] = (half)-1.0 * points_acc.x[k];
//                 }
//
//                 // Compute the distance to the point in each dimension
//                 // Accumulate in C instead of using another fragment
//                 wmma::mma_sync(points_acc, point_frag, i_frag, points_acc);
//
//                 // Store this intermediate result into shared memory, considering the other warps
//                 wmma::store_matrix_sync(sharedTmpArrayHalf + sharedMemoryTmpOffset, points_acc, TILE_SIZE, wmma::mem_row_major);
//
//                 // Create a new fragment in col_major for the multiplication and load it from shared memory
//                 wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> points_acc_col;
//                 wmma::load_matrix_sync(points_acc_col, sharedTmpArrayHalf + sharedMemoryPointOffset, TILE_SIZE);
//                 // Reuse the previous fragment that is in row major
//                 wmma::load_matrix_sync(point_frag, sharedTmpArrayHalf + sharedMemoryPointOffset, TILE_SIZE);
//
//                 // Accumulate in the single precision fragment, adding the dimensions with each loop iteration
//                 // We reuse the points_acc fragment from before instead of creating a new one and loading it
//                 wmma::mma_sync(tmp_dist_acc, point_frag, points_acc_col, tmp_dist_acc);
//             }
//
//             // tmp_dist_acc contains the distance accross all the dimensions, so store it into shared memory first
//             wmma::store_matrix_sync(sharedTmpArray + sharedMemoryTmpOffset, tmp_dist_acc, TILE_SIZE, wmma::mem_row_major);
//
//             // But only keep the real distance between the points, i.e., the diagonals
//             if (warp.thread_rank() < TILE_SIZE)
//             {
//                 // Hopefully, not too many bank conflicts
//                 // result[warpId * pointsPerWarp * nbQueries + i * nbQueries + j + warp.thread_rank()] =
//                 //         sharedTmpArray[sharedMemoryTmpOffset + warp.thread_rank() * TILE_SIZE + warp.thread_rank()];
//
//                 float resultDistance = sharedTmpArray[sharedMemoryTmpOffset + warp.thread_rank() * TILE_SIZE + warp.thread_rank()];
//
// 				#if ACCUM_PREC == 16
// 				if (hsqrt(resultDistance) <= (*epsilon))
// 				#else
// 				if (sqrt(resultDistance) <= (*epsilon))
// 				#endif
// 				{
// 					atomicAdd(nbNeighbors, int(1));
// 				}
//             }
//             warp.sync();
//         }
//     }
// }



__global__ void distanceCalculationBruteForceCuda(
	half* database,
	unsigned int* nbQueries,
	unsigned int* queryOffset,
	ACCUM_TYPE* epsilon,
	unsigned int* nbNeighbors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((*nbQueries) <= tid)
	{
		return;
	}

	half point[INPUT_DATA_DIM];
	for (int i = 0; i < INPUT_DATA_DIM; ++i)
	{
		point[i] = database[tid * COMPUTE_DIM + i];
	}

	for (int i = 0; i < (*nbQueries); ++i)
	{
		ACCUM_TYPE resultDistance = 0.0;

		for (int j = 0; j < INPUT_DATA_DIM; ++j)
		{
			resultDistance += (ACCUM_TYPE)((point[j] * database[i * COMPUTE_DIM + j]) - (point[j] * database[i * COMPUTE_DIM + j]));
		}

		#if ACCUM_PREC == 16
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
	ACCUM_TYPE* epsilon,
	unsigned int* nbNeighbors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((*nbQueries) <= tid)
	{
		return;
	}

	half2 point[INPUT_DATA_DIM / 2];
	for (int i = 0; i < INPUT_DATA_DIM / 2; ++i)
	{
		point[i] = database[tid * (INPUT_DATA_DIM / 2) + i];
	}

	for (int i = 0; i < (*nbQueries); ++i)
	{
		#if ACCUM_PREC == 16
			half resultDistance = 0.0;
			for (int j = 0; j < INPUT_DATA_DIM / 2; ++j)
			{
				half2 tmpResult = __hsub2(__hmul2(point[j], database[i * (INPUT_DATA_DIM / 2) + i]), __hmul2(point[j], database[i * (INPUT_DATA_DIM / 2) + i]));
				resultDistance += __lo2half(tmpResult) + __high2half(tmpResult);
			}
		#else
			float resultDistance = 0.0;
			for (int j = 0; j < INPUT_DATA_DIM / 2; ++j)
			{
				half2 tmpResult = __hsub2(__hmul2(point[j], database[i * (INPUT_DATA_DIM / 2) + i]), __hmul2(point[j], database[i * (INPUT_DATA_DIM / 2) + i]));
				resultDistance += __low2float(tmpResult) + __high2float(tmpResult);
			}
		#endif

		#if ACCUM_PREC == 16
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
	ACCUM_TYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> identityFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half> firstStepAccumulator;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, ACCUM_TYPE> secondStepAccumulator;

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
				ACCUM_TYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if ACCUM_PREC == 16
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
	ACCUM_TYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> identityFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half> firstStepAccumulator;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, ACCUM_TYPE> secondStepAccumulator;

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
				ACCUM_TYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if ACCUM_PREC == 16
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
	ACCUM_TYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayTmpFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, ACCUM_TYPE> secondStepAccumulator;

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
				ACCUM_TYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if ACCUM_PREC == 16
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
	ACCUM_TYPE* epsilon,
	unsigned int* nbNeighbors)
{
	__shared__ half sharedArrayQueryPoint[WARP_PER_BLOCK * COMPUTE_DIM];
	__shared__ half sharedArrayTmpFirstStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];
	__shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int warpId = tid / WARP_SIZE;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, ACCUM_TYPE> secondStepAccumulator;

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
				ACCUM_TYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

				#if ACCUM_PREC == 16
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



//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\



__global__ void distanceCalculationGridCuda(
    unsigned int* batchBegin,
    unsigned int* batchSize,
    ACCUM_TYPE* database,
    unsigned int* originPointIndex,
    ACCUM_TYPE* epsilon,
    struct grid* grid,
    unsigned int* gridLookupArr,
    struct gridCellLookup* gridCellLookupArr,
    ACCUM_TYPE* minArr,
    unsigned int* nCells,
    unsigned int* cnt,
    unsigned int* nNonEmptyCells,
    int* pointIDKey,
    int* pointInDistVal)
{
    unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if ((*batchSize) <= tid)
    {
        return;
    }

    // Get the next query point in the "local" queue
    unsigned int pointId = atomicAdd(batchBegin, int(1));

    COMPUTE_TYPE point[INPUT_DATA_DIM];
    for (int i = 0; i < INPUT_DATA_DIM; ++i)
    {
        point[i] = database[ originPointIndex[pointId] * COMPUTE_DIM + i ];
    }

    // Calculate the coords of the Cell for the point and the min/max ranges in each dimension
	unsigned int nDCellIDs[INDEXED_DIM];
    unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];

    for (int i = 0; i < INDEXED_DIM; ++i)
    {
        nDCellIDs[i] = ((ACCUM_TYPE)point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); // Boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); // Boundary conditions (don't go beyond the maximum number of cells)
    }

    unsigned int indexes[INDEXED_DIM];
    unsigned int loopRng[INDEXED_DIM];

    for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < INDEXED_DIM; x++)
			{
				indexes[x] = loopRng[x];
			}

			evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, grid,
					gridLookupArr, point, cnt, pointIDKey, pointInDistVal, originPointIndex[pointId], nDCellIDs);

		} //end loop body
}



// Only handle half precision computation for the moment (and accumulation in half or single precision)
__global__ void distanceCalculationGridTensor(
    unsigned int* batchBegin,
    unsigned int* batchSize,
    ACCUM_TYPE* database,
    unsigned int* originPointIndex,
    ACCUM_TYPE* epsilon,
    struct grid* grid,
    unsigned int* gridLookupArr,
    struct gridCellLookup* gridCellLookupArr,
    ACCUM_TYPE* minArr,
    unsigned int* nCells,
    unsigned int* cnt,
    unsigned int* nNonEmptyCells,
    int* pointIDKey,
    int* pointInDistVal,
    COMPUTE_TYPE* identityMatrix)
{
	// Store the current query point of a warp
	__shared__ half sharedArrayQueryPoints[COMPUTE_DIM * WARP_PER_BLOCK];
    // Store the result of the first step (distance between the individual coordinates)
    __shared__ half sharedArrayFirstStep[TILE_SIZE_HALF * TILE_SIZE_HALF * WARP_PER_BLOCK];
    // Store the result of the second step (actual distance between points)
    __shared__ float sharedArrayResSecondStep[TILE_SIZE_HALF * TILE_SIZE_HALF * WARP_PER_BLOCK];

    thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpId = threadIdx.x / WARP_SIZE;

	unsigned int sharedMemoryOffset = warpId * TILE_SIZE_HALF * TILE_SIZE_HALF;

	unsigned int nDCellIDs[INDEXED_DIM];
	unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];

	unsigned int indexes[INDEXED_DIM];
	unsigned int loopRng[INDEXED_DIM];

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> pointsFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> candidatesFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, float> distanceAccumulatorFragment;

	// For each point to compute per warp
	for (int i = 0; i < POINTS_PER_WARP; ++i)
	{
		// Thread 0 of the warp gets the next point to compute, and shares it with the rest of the warp
		unsigned int pointId;
		if (0 == warp.thread_rank())
		{
			pointId = atomicAdd(batchBegin, int(1));
		}
		pointId = __shfl_sync(0xffffffff, pointId, 0);

		// Copy the query point into shared memory
		unsigned int nbDimToCopy = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbDimToCopy; ++j)
		{
			if (warp.thread_rank() * nbDimToCopy < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[warpId * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j] =
					(half)database[ originPointIndex[pointId] * COMPUTE_DIM + warp.thread_rank() * nbDimToCopy + j ];
			}
		}
		warp.sync();

		// Compute the neighboring cells in all indexeded dimensions
		for (int j = 0; j < INDEXED_DIM; ++j)
		{
			nDCellIDs[j] = ((float)sharedArrayQueryPoints[warpId * COMPUTE_DIM + j] - minArr[j]) / (*epsilon);
			nDMinCellIDs[j] = max(0, nDCellIDs[j] - 1);
			nDMaxCellIDs[j] = min(nCells[j] - 1, nDCellIDs[j] + 1);
		}

		// wmma::load_matrix_sync(identityFragment, identityMatrix, TILE_SIZE);

		float resultDistance = 0.0;

		// Iterate over the neighboring cells
		for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
			for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
			#include "kernelloops.h"
			{ //beginning of loop body
				for (int x = 0; x < INDEXED_DIM; ++x)
				{
					indexes[x] = loopRng[x];
				}

				uint64_t cellLinearId = getLinearID_nDimensionsGPUKernel(indexes, nCells, INDEXED_DIM);
				struct gridCellLookup tmp;
				tmp.gridLinearID = cellLinearId;

				// Find if the neighboring cell is empty or not
				if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
				{
					struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
						gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
					unsigned int gridIndex = resultBinSearch->idx;

					// Iterate over the points in the neighboring cell
					// Compute TILE_SIZE candidates by TILE_SIZE (by default, 16 by 16)
					unsigned int nbCandidatesLeft = grid[gridIndex].indexmax - grid[gridIndex].indexmin + 1;
					for (int k = grid[gridIndex].indexmin; k <= grid[gridIndex].indexmax; k += TILE_SIZE_HALF)
					{
						// Set the result distance fragment to 0 for these candidate points
						wmma::fill_fragment(distanceAccumulatorFragment, 0.0f);

						unsigned int candidateId = gridLookupArr[k + (warp.thread_rank() / 2)];

						// Iterate over the number of steps needed to compute the distance in all dimensions
						// Reminder: COMPUTE_DIM is always a multiple of TILE_SIZE
						for (int n = 0; n < COMPUTE_DIM / TILE_SIZE_HALF; ++n)
						{
							// Load the query point from shared memory into the fragment
							// wmma::load_matrix_sync(pointsFragment, sharedArrayQueryPoints + warpId * WARP_SIZE + n * TILE_SIZE, 0);

							// Page the candidate points into shared memory
							// 16 points in 16 dimensions, so a thread pages 8 dimensions of a point
							// Necessary since the candidate are not always coalesced into global memory
							// Directly compute the difference between the query point and the candidate points coordinates
							unsigned int dimOffset = (TILE_SIZE_HALF * TILE_SIZE_HALF) / WARP_SIZE;
							for (int l = 0; l < dimOffset; ++l)
							{
								// sharedArrayFirstStep[sharedMemoryOffset + warp.thread_rank() * dimOffset + l] =
								// 	(COMPUTE_TYPE)(-1.0f) * database[candidateId * COMPUTE_DIM + n * TILE_SIZE + warp.thread_rank() * dimOffset + l];
								sharedArrayFirstStep[sharedMemoryOffset + warp.thread_rank() * dimOffset + l]
									= sharedArrayQueryPoints[warpId * COMPUTE_DIM + n * TILE_SIZE_HALF + (warp.thread_rank() % 2) * dimOffset + l]
									- (half)database[candidateId * COMPUTE_DIM + n * TILE_SIZE_HALF + (warp.thread_rank() % 2) * dimOffset + l];
							}
							warp.sync();

							// Load the candidate points from shared memory into the fragment
							// wmma::load_matrix_sync(tmpAccumulator, sharedArrayFirstStep + sharedMemoryOffset, wmma::mem_row_major);

							// Compute the difference between the coordinates of the query point and the candidate points
							// Use the identity matrix to keep the coordinates of the query point untouched
							// I.e., compute pointsFragment - tmpAccumulator
							// wmma::mma_sync(tmpAccumulator, pointsFragment, identityFragment, tmpAccumulator);

							// Load the difference between the query point and the candidate points coordinates
							// from shared memory into the fragments
							wmma::load_matrix_sync(pointsFragment, sharedArrayFirstStep + sharedMemoryOffset, TILE_SIZE_HALF);
							wmma::load_matrix_sync(candidatesFragment, sharedArrayFirstStep + sharedMemoryOffset, TILE_SIZE_HALF);

							// Accumulate into distanceAccumulatorFragment the running distance between the query point and the candidate points
							wmma::mma_sync(distanceAccumulatorFragment, pointsFragment, candidatesFragment, distanceAccumulatorFragment);

							#if SHORT_CIRCUIT
								// Store the distance between the query point and the candidate points into shared memory
								wmma::store_matrix_sync(sharedArrayResSecondStep + sharedMemoryOffset, distanceAccumulatorFragment, TILE_SIZE_HALF, wmma::mem_row_major);

								// Extract the distance between the query point and the candidate points
								// These 16 distances are located in the diagonal of the matrix
								int threadsShortCircuit = 0;
								if (warp.thread_rank() < TILE_SIZE && warp.thread_rank() < nbCandidatesLeft)
								{
									resultDistance = sharedArrayResSecondStep[sharedMemoryOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

									// Check if this thread should short-circuit
									int shortCircuit = 0;
									#if ACCUM_PREC == 16
									if (hsqrt(resultDistance) > (*epsilon))
									#else
									if (sqrt(resultDistance) > (*epsilon))
									#endif
									{
										shortCircuit = 1;
									}

									// int nbThreadsShortCircuit = 0;
									// Match if all 16 candidate points short-circuited
									__match_all_sync(__activemask(), shortCircuit, &threadsShortCircuit);
									// If the 16 candidates are beyond epsilon, then break this loop
									// if (nbThreadsShortCircuit)
									// {
									// 	n = COMPUTE_DIM;
									// }
								}

								// Get from thread 0 if the threads that computed the distances short-circuited
								threadsShortCircuit = __shfl_sync(0xffffffff, threadsShortCircuit, 0);
								if (threadsShortCircuit)
								{
									n = COMPUTE_DIM;
								}
							#endif
						} // For steps over the dimensions

						#if SHORT_CIRCUIT
							if (warp.thread_rank() < TILE_SIZE_HALF && warp.thread_rank() < nbCandidatesLeft)
							{
								// The distance was already computed on the last short-circuit check
								#if ACCUM_PREC == 16
								if (hsqrt(resultDistance) <= (*epsilon))
								#else
								if (sqrt(resultDistance) <= (*epsilon))
								#endif
								{
									unsigned int tmpIdx = atomicAdd(cnt, int(1));
									pointIDKey[tmpIdx] = originPointIndex[pointId];
									pointInDistVal[tmpIdx] = gridLookupArr[k];
								}
							}
						#else
							// Store the distance between the query point and the candidate points into shared memory
							wmma::store_matrix_sync(sharedArrayResSecondStep + sharedMemoryOffset, distanceAccumulatorFragment, TILE_SIZE_HALF, wmma::mem_row_major);

							// Extract the distance between the query point and the candidate points
							// These 16 distances are located in the diagonal of the matrix
							if (warp.thread_rank() < TILE_SIZE_HALF && warp.thread_rank() < nbCandidatesLeft)
							{
								float resultDistance = sharedArrayResSecondStep[sharedMemoryOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

								// Discriminate the two cases, as a different function should be used depending on the precision
								#if ACCUM_PREC == 16
								if (hsqrt(resultDistance) <= (*epsilon))
								#else
								if (sqrt(resultDistance) <= (*epsilon))
								#endif
								{
									unsigned int tmpIdx = atomicAdd(cnt, int(1));
									pointIDKey[tmpIdx] = originPointIndex[pointId];
									pointInDistVal[tmpIdx] = gridLookupArr[k];
								}
							}
						#endif

						nbCandidatesLeft -= TILE_SIZE_HALF;
					} // For candidates in the cell
				} // If the neighoring candidate cell is not empty
			} // End loop body
	} // For the number of query points assigned to this warp
} // Kernel



__global__ void distanceCalculationGridTensor_TwoStepsComputePagingOneQuery(
	unsigned int* batchBegin,
	unsigned int* batchSize,
	half* database,
	unsigned int* originPointIndex,
	half* identityMatrix,
	ACCUM_TYPE* epsilon,
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
	__shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> identityFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half> firstStepAccumulator;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, ACCUM_TYPE> secondStepAccumulator;

	wmma::load_matrix_sync(identityFragment, identityMatrix, TILE_SIZE_HALF);

	unsigned int nDCellIDs[INDEXED_DIM];
	unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];
	unsigned int indexes[INDEXED_DIM];
	unsigned int loopRng[INDEXED_DIM];

	unsigned int firstQueryId;
	if (0 == warp.thread_rank())
	{
		firstQueryId = atomicAdd(batchBegin, int(POINTS_PER_WARP));
	}
	firstQueryId = __shfl_sync(0xffffffff, firstQueryId, 0);

	for (int i = firstQueryId; i < (firstQueryId + POINTS_PER_WARP); ++i)
	{
		unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbDimsToPage; ++j)
		{
			if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j] =
					database[ originPointIndex[i] * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j];
			}
		}

		for (int j = 0; j < INDEXED_DIM; ++j)
		{
			nDCellIDs[j] = (sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j] - minArr[j]) / (half)(*epsilon);
			nDMinCellIDs[j] = max(0, nDCellIDs[j] - 1);
			nDMaxCellIDs[j] = min(nCells[j] - 1, nDCellIDs[j] + 1);
		}

		ACCUM_TYPE resultDistance = 0.0;

		for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
			for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
			#include "kernelloops.h"
			{ //beginning of loop body
				for (int x = 0; x < INDEXED_DIM; ++x)
				{
					indexes[x] = loopRng[x];
				}

				uint64_t cellLinearId = getLinearID_nDimensionsGPUKernel(indexes, nCells, INDEXED_DIM);
				struct gridCellLookup tmp;
				tmp.gridLinearID = cellLinearId;

				// Find if the neighboring cell is empty or not
				if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
				{
					struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
						gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
					unsigned int gridIndex = resultBinSearch->idx;

					for (int k = grid[gridIndex].indexmin; k <= grid[gridIndex].indexmax; k += TILE_SIZE_HALF)
					{
						unsigned int nbCandidatesLeft = grid[gridIndex].indexmax - k + 1;

						wmma::fill_fragment(secondStepAccumulator, 0.0f);

						unsigned int candidateId = gridLookupArr[k + (warp.thread_rank() / 2)];

						for (int n = 0; n < COMPUTE_DIM; n += TILE_SIZE_HALF)
						{
							wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoints + (warpIdInBlock * COMPUTE_DIM + k), 0);

							// unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
							for (int j = 0; j < nbDimsToPage; ++j)
							{
								if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
								{
									sharedArrayResultFirstStep[sharedArrayResultOffset + warp.thread_rank() * nbDimsToPage + j] =
										database[candidateId * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j];
								}
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
						}

						wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

						if (warp.thread_rank() < TILE_SIZE_HALF)
						{
							resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

							#if ACCUM_PREC == 16
							if(hsqrt(resultDistance) <= (*epsilon))
							#else
							if(sqrt(resultDistance) <= (*epsilon))
							#endif
							{
								unsigned int tmpIdx = atomicAdd(cnt, int(1));
								pointIDKey[tmpIdx] = originPointIndex[i];
								pointInDistVal[tmpIdx] = candidateId;
							}
						}
						warp.sync();
					}
				}
			}
	}
}



__global__ void distanceCalculationGridTensor_OneStepComputePagingOneQuery(
	unsigned int* batchBegin,
	unsigned int* batchSize,
	half* database,
	unsigned int* originPointIndex,
	ACCUM_TYPE* epsilon,
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
	__shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * TILE_SIZE_HALF * TILE_SIZE_HALF];

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;
	unsigned int sharedArrayResultOffset = warpIdInBlock * TILE_SIZE_HALF * TILE_SIZE_HALF;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

	wmma::fragment<wmma::matrix_a, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::row_major> matrixAFragment;
	wmma::fragment<wmma::matrix_b, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, half, wmma::col_major> matrixBFragment;
	wmma::fragment<wmma::accumulator, TILE_SIZE_HALF, TILE_SIZE_HALF, TILE_SIZE_HALF, ACCUM_TYPE> secondStepAccumulator;

	unsigned int nDCellIDs[INDEXED_DIM];
	unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];
	unsigned int indexes[INDEXED_DIM];
	unsigned int loopRng[INDEXED_DIM];

	unsigned int firstQueryId;
	if (0 == warp.thread_rank())
	{
		firstQueryId = atomicAdd(batchBegin, int(POINTS_PER_WARP));
	}
	firstQueryId = __shfl_sync(0xffffffff, firstQueryId, 0);

	for (int i = firstQueryId; i < (firstQueryId + POINTS_PER_WARP); ++i)
	{
		unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
		for (int j = 0; j < nbDimsToPage; ++j)
		{
			if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j] =
					database[ originPointIndex[i] * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j];
			}
		}

		for (int j = 0; j < INDEXED_DIM; ++j)
		{
			nDCellIDs[j] = (sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j] - minArr[j]) / (half)(*epsilon);
			nDMinCellIDs[j] = max(0, nDCellIDs[j] - 1);
			nDMaxCellIDs[j] = min(nCells[j] - 1, nDCellIDs[j] + 1);
		}

		ACCUM_TYPE resultDistance = 0.0;

		for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
			for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
			#include "kernelloops.h"
			{ //beginning of loop body
				for (int x = 0; x < INDEXED_DIM; ++x)
				{
					indexes[x] = loopRng[x];
				}

				uint64_t cellLinearId = getLinearID_nDimensionsGPUKernel(indexes, nCells, INDEXED_DIM);
				struct gridCellLookup tmp;
				tmp.gridLinearID = cellLinearId;

				// Find if the neighboring cell is empty or not
				if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
				{
					struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
						gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
					unsigned int gridIndex = resultBinSearch->idx;

					for (int k = grid[gridIndex].indexmin; k <= grid[gridIndex].indexmax; k += TILE_SIZE_HALF)
					{
						unsigned int nbCandidatesLeft = grid[gridIndex].indexmax - k + 1;

						wmma::fill_fragment(secondStepAccumulator, 0.0f);

						unsigned int candidateId = gridLookupArr[k + (warp.thread_rank() / 2)];

						for (int n = 0; n < COMPUTE_DIM; n += TILE_SIZE_HALF)
						{
							wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoints + (warpIdInBlock * COMPUTE_DIM + k), 0);

							// unsigned int nbDimsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
							for (int j = 0; j < nbDimsToPage; ++j)
							{
								if ((warp.thread_rank() * nbDimsToPage + j) < COMPUTE_DIM)
								{
									sharedArrayResultFirstStep[sharedArrayResultOffset + warp.thread_rank() * nbDimsToPage + j] =
										database[candidateId * COMPUTE_DIM + warp.thread_rank() * nbDimsToPage + j] * (half)-1.0;
								}
							}
							wmma::load_matrix_sync(matrixBFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, TILE_SIZE_HALF);

							wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
						}

						wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, TILE_SIZE_HALF, wmma::mem_row_major);

						if (warp.thread_rank() < TILE_SIZE_HALF)
						{
							resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * TILE_SIZE_HALF + warp.thread_rank()];

							#if ACCUM_PREC == 16
							if(hsqrt(resultDistance) <= (*epsilon))
							#else
							if(sqrt(resultDistance) <= (*epsilon))
							#endif
							{
								unsigned int tmpIdx = atomicAdd(cnt, int(1));
								pointIDKey[tmpIdx] = originPointIndex[i];
								pointInDistVal[tmpIdx] = candidateId;
							}
						}
						warp.sync();
					}
				}
			}
	}
}

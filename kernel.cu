#include "kernel.h"
#include "structs.h"
#include "params.h"

#include <math.h>

#include <cooperative_groups.h>

#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

//namespace cg = cooperative_groups;
using namespace cooperative_groups;



__device__ void print(unsigned int tid, unsigned int value)
{
		if(0 == tid)
		{
				printf("threadIdx.x 0, value = %d\n", value);
		}
}



/******************************************************************************/



__global__ void kernelIndexComputeNonemptyCells(
	DTYPE * database,
	unsigned int * N,
	DTYPE * epsilon,
	DTYPE * minArr,
	unsigned int * nCells,
	uint64_t * pointCellArr,
	unsigned int * databaseVal,
	bool enumerate)
{
		unsigned int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

		if(*N <= tid)
		{
				return;
		}

		unsigned int pointID = tid * GPUNUMDIM;

		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j = 0; j < NUMINDEXEDDIM; j++)
		{
				tmpNDCellIdx[j] = ((database[pointID + j] - minArr[j]) / (*epsilon));
		}
		uint64_t linearID = getLinearID_nDimensionsGPU(tmpNDCellIdx, nCells, NUMINDEXEDDIM);

		pointCellArr[tid] = linearID;

		if(enumerate)
		{
				databaseVal[tid] = tid;
		}
}



/******************************************************************************/



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
		schedulingCell * sortedCells)
{

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if(*nNonEmptyCells <= tid)
		{
				return;
		}

		unsigned int cell = gridCellLookupArr[tid].idx;
		unsigned int nbNeighborPoints = 0;
		unsigned int tmpId = indexLookupArr[ index[cell].indexmin ];

		DTYPE point[GPUNUMDIM];
		for(int i = 0; i < GPUNUMDIM; ++i)
		{
				point[i] = database[tmpId * GPUNUMDIM + i];
		}

		unsigned int nDCellIDs[NUMINDEXEDDIM];

		unsigned int nDMinCellIDs[NUMINDEXEDDIM];
		unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

		for(int n = 0; n < NUMINDEXEDDIM; n++)
		{
				nDCellIDs[n] = (point[n] - minArr[n]) / (*epsilon);
				nDMinCellIDs[n] = max(0, nDCellIDs[n] - 1);;
				nDMaxCellIDs[n] = min(nCells[n] - 1, nDCellIDs[n] + 1);
		}

		unsigned int indexes[NUMINDEXEDDIM];
		unsigned int loopRng[NUMINDEXEDDIM];

		// for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		// 	for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{
				for (int x = 0; x < NUMINDEXEDDIM; x++){
						indexes[x] = loopRng[x];
				}

				uint64_t cellID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
				struct gridCellLookup tmp;
				tmp.gridLinearID = cellID;
				if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
				{
						struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
						unsigned int GridIndex = resultBinSearch->idx;
						nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
				}
		}

		sortedCells[tid].nbPoints = nbNeighborPoints;
		sortedCells[tid].cellId = cell;
}



/******************************************************************************/


//TODO use the unicomp pattern
//TODO modify to use the new GPU index
__global__ void sortByWorkLoadUnicomp(
		DTYPE * database,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		schedulingCell * sortedCells)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(*nNonEmptyCells <= tid)
	{
		return;
	}

	int cell = gridCellLookupArr[tid].idx;
	int nbNeighborPoints = 0;
	int tmpId = indexLookupArr[ index[cell].indexmin ];

	DTYPE point[NUMINDEXEDDIM];

	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

	for(int n = 0; n < NUMINDEXEDDIM; n++)
	{
		point[n] = database[tmpId * NUMINDEXEDDIM + n];
		nDCellIDs[n] = (point[n] - minArr[n]) / (*epsilon);
		nDMinCellIDs[n] = max(0, nDCellIDs[n] - 1);;
		nDMaxCellIDs[n] = min(nCells[n] - 1, nDCellIDs[n] + 1);
	}

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{
			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t cellID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			struct gridCellLookup tmp;
			tmp.gridLinearID = cellID;
			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;
				nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
			}

		}

	sortedCells[tid].nbPoints = nbNeighborPoints;
	sortedCells[tid].cellId = cell;

}



/******************************************************************************/


//TODO modify to use the new GPU index
__global__ void sortByWorkLoadLidUnicomp(
		DTYPE* database,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		schedulingCell * sortedCells)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(*nNonEmptyCells <= tid)
	{
		return;
	}

	int cell = gridCellLookupArr[tid].idx;
	int nbNeighborPoints = 0;
	int tmpId = indexLookupArr[ index[cell].indexmin ];

	DTYPE point[NUMINDEXEDDIM];

	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

	for(int n = 0; n < NUMINDEXEDDIM; n++)
	{
		point[n] = database[tmpId * NUMINDEXEDDIM + n];
		nDCellIDs[n] = (point[n] - minArr[n]) / (*epsilon);
		nDMinCellIDs[n] = max(0, nDCellIDs[n] - 1);;
		nDMaxCellIDs[n] = min(nCells[n] - 1, nDCellIDs[n] + 1);
	}

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (int x = 0; x < NUMINDEXEDDIM; x++){
		indexes[x] = nDCellIDs[x];
	}

	uint64_t originCellID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{
			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t cellID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			if(originCellID <= cellID)
			{
				struct gridCellLookup tmp;
				tmp.gridLinearID = cellID;
				if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
				{
					struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
					unsigned int GridIndex = resultBinSearch->idx;
					nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
				}
			}

		}

	sortedCells[tid].nbPoints = nbNeighborPoints;
	sortedCells[tid].cellId = cell;

}



/******************************************************************************/



__device__ uint64_t getLinearID_nDimensionsGPU(
		unsigned int * indexes,
		unsigned int * dimLen,
		unsigned int nDimensions)
{
    uint64_t offset = 0;
	uint64_t multiplier = 1;

	for (int i = 0; i < nDimensions; i++)
	{
		offset += (uint64_t) indexes[i] * multiplier;
		multiplier *= dimLen[i];
	}

	return offset;
}



/******************************************************************************/



__forceinline__ __device__ void evalPoint(
		unsigned int * indexLookupArr,
		int k,
		DTYPE * database,
		DTYPE * epsilon,
		DTYPE * point,
		unsigned int * cnt,
		int * pointIDKey,
		int * pointInDistVal,
		int pointIdx,
		bool differentCell)
{
	// unsigned int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

	DTYPE runningTotalDist = 0;
	unsigned int dataIdx = indexLookupArr[k];

	// if (threadIdx.x == 0) {
	// 	# if __CUDA_ARCH__>=200
	// 	printf("BLOCK %d COMPARING %d to %d\n", blockIdx.x, pointIdx, dataIdx);
	// 	#endif
	// }

	for(int l = 0; l < GPUNUMDIM; l++){
		runningTotalDist += ( database[dataIdx * GPUNUMDIM + l] - point[l])
				* (database[dataIdx * GPUNUMDIM + l] - point[l] );
	}

	if(sqrt(runningTotalDist) <= (*epsilon)){
	//if(runningTotalDist <= ((*epsilon) * (*epsilon))){
		unsigned int idx = atomicAdd(cnt, int(1));
		// pointIDKey[idx] = pointIdx; // --> HERE
		// pointInDistVal[idx] = dataIdx;

		if(differentCell)
		{
			unsigned int idx = atomicAdd(cnt, int(1));
			// pointIDKey[idx] = dataIdx;
			// // pointIDKey[tid] = dataIdx;
			// pointInDistVal[idx] = pointIdx;
			// // pointInDistVal[tid] = pointIdx;
		}
	}
}



/******************************************************************************/



__device__ void evaluateCell(
		unsigned int * nCells,
		unsigned int * indexes,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int * nNonEmptyCells,
		DTYPE * database,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		DTYPE * point,
		unsigned int * cnt,
		int * pointIDKey,
		int * pointInDistVal,
		int pointIdx,
		bool differentCell,
		unsigned int * nDCellIDs)
{
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;


		for(int k = index[GridIndex].indexmin; k <= index[GridIndex].indexmax; k+=BLOCKSIZE) {
			uint64_t pointToCompare = (k + threadIdx.x);
			if (!(pointToCompare > index[GridIndex].indexmax)) {
				evalPoint(indexLookupArr, pointToCompare, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx, differentCell);
			}
		}
	}
}



/******************************************************************************/



__forceinline__ __device__ void evalPointUnicompOrigin(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx)
{
	DTYPE runningTotalDist = 0;
	unsigned int dataIdx = indexLookupArr[k];

	for (int l = 0; l < GPUNUMDIM; l++)
	{
		runningTotalDist += (database[dataIdx * GPUNUMDIM + l] - point[l]) * (database[dataIdx * GPUNUMDIM + l] - point[l]);
	}

	if (sqrt(runningTotalDist) <= (*epsilon)){
	//if(runningTotalDist <= ((*epsilon) * (*epsilon))){
		unsigned int idx = atomicAdd(cnt, int(1));
		// assert(idx < 2000000);
		pointIDKey[idx] = pointIdx; // --> HERE
		pointInDistVal[idx] = dataIdx;
	}
}



/******************************************************************************/



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
		unsigned int numThread)
{
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		int begin = index[GridIndex].indexmin;
		int end = index[GridIndex].indexmax;
		int nbElem = end - begin + 1;
		if(numThread < nbElem)
		{
			int size = nbElem / nbThreads;
			int oneMore = nbElem - (size * nbThreads);
			if(nbElem == (size * nbThreads))
			{
				begin += size * numThread;
				end = begin + size - 1;
			}else{
				begin += numThread * size + ((numThread < oneMore)?numThread:oneMore);
				end = begin + size - 1 + (numThread < oneMore);
			}

			for(int k = begin; k <= end; k++)
			{
				evalPointUnicompOrigin(indexLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			}
		}
	}
}



/******************************************************************************/



__forceinline__ __device__ void evalPointUnicompAdjacent(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx)
{
	DTYPE runningTotalDist = 0;
	unsigned int dataIdx = indexLookupArr[k];

	for (int l = 0; l < GPUNUMDIM; l++)
	{
		runningTotalDist += (database[dataIdx * GPUNUMDIM + l] - point[l]) * (database[dataIdx * GPUNUMDIM + l] - point[l]);
	}

	if (sqrt(runningTotalDist) <= (*epsilon)){
	//if(runningTotalDist <= ((*epsilon) * (*epsilon))){
		unsigned int idx = atomicAdd(cnt, int(2));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
		pointIDKey[idx + 1] = dataIdx;
		pointInDistVal[idx + 1] = pointIdx;
	}
}



/******************************************************************************/



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
		unsigned int numThread)
{
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		int begin = index[GridIndex].indexmin;
		int end = index[GridIndex].indexmax;
		int nbElem = end - begin + 1;
		if(numThread < nbElem)
		{
			int size = nbElem / nbThreads;
			int oneMore = nbElem - (size * nbThreads);
			if(nbElem == (size * nbThreads))
			{
				begin += size * numThread;
				end = begin + size - 1;
			}else{
				begin += numThread * size + ((numThread < oneMore)?numThread:oneMore);
				end = begin + size - 1 + (numThread < oneMore);
			}

			for(int k = begin; k <= end; k++)
			{
				evalPointUnicompAdjacent(indexLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			}
		}
	}
}



/******************************************************************************/



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
		unsigned int * estimatedResult)
{

	unsigned int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

	if((*N) <= tid)
	{
		return;
	}

	unsigned int pointID = tid  * (*sampleOffset);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; ++i)
	{
		#if SORT_BY_WORKLOAD
			point[i] = database[ originPointIndex[pointID] * GPUNUMDIM + i ];
		#else
			point[i] = database[ pointID * GPUNUMDIM + i ];
		#endif
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; ++i)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)
	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	unsigned int localNeighborCounter = 0;

	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; ++x)
			{
				indexes[x] = loopRng[x];
			}

			uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
			//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

			struct gridCellLookup tmp;
			tmp.gridLinearID = calcLinearID;

			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				//in the GPU implementation we go directly to computing neighbors so that we don't need to
				//store a buffer of the cells to check
				//cellsToCheck->push_back(calcLinearID);

				//HERE WE COMPUTE THE NEIGHBORS FOR THE CELL
				//XXXXXXXXXXXXXXXXXXXXXXXXX

				struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;

				for (int k = index[GridIndex].indexmin; k <= index[GridIndex].indexmax; ++k)
				{
					DTYPE runningTotalDist = 0;
					unsigned int dataIdx = indexLookupArr[k];

					for (int l = 0; l < GPUNUMDIM; ++l)
					{
						runningTotalDist += (database[dataIdx * GPUNUMDIM + l]  - point[l])
								* (database[dataIdx * GPUNUMDIM + l] - point[l]);
					}

					if (sqrt(runningTotalDist) <= (*epsilon))
					{
						unsigned int idx = atomicAdd(cnt, int(1));
						localNeighborCounter++;
					}
				}
			}
		} //end loop body

	estimatedResult[tid] = localNeighborCounter;

}



/******************************************************************************/



// __device__ int counter = 0;

// Global memory kernel - Initial version ("GPU")
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
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x);

	if (*N <= (tid / BLOCKSIZE))
	{
		return;
	}

	unsigned int pointId = (*batchBegin) + blockIdx.x;
	// if (threadIdx.x == 0) {
	// 	# if __CUDA_ARCH__>=200
	// 	printf("BLOCK %d COMPUTING %d\n", blockIdx.x, pointId);
	// 	#endif
	// }
	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++)
	{
		#if SORT_BY_WORKLOAD
			point[i] = database[ originPointIndex[pointId] * GPUNUMDIM + i ];
		#else
			point[i] = database[ pointId * GPUNUMDIM + i ];
		#endif
	}

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		# if __CUDA_ARCH__>=200
		printf("Examining Point: %d(%d)\n", pointId, originPointIndex[pointId]);
		#endif
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	// unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	// unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)
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

			evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index,
					indexLookupArr, point, cnt, pointIDKey, pointInDistVal, originPointIndex[pointId], false, nDCellIDs);

		} //end loop body

}

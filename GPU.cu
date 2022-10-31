//precompute direct neighbors with the GPU:
#include "GPU.h"
#include "kernel.h"
#include "SortByWorkload.h"
#include "structs.h"
#include "params.h"
#include "WorkQueue.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <unistd.h>
#include "omp.h"

#include <pthread.h>

// //thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> // for streams for thrust (added with Thrust v1.8)
//
// //for warming up GPU:
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

//elements for the result set
//FOR A SINGLE KERNEL INVOCATION
//NOT FOR THE BATCHED ONE
#define BUFFERELEM 300000000 //400000000-original (when removing the data from the device before putting it back for the sort)

//FOR THE BATCHED EXECUTION:
//#define BATCHTOTALELEM 1200000000 //THE TOTAL SIZE ALLOCATED ON THE HOST
//THE NUMBER OF BATCHES AND THE SIZE OF THE BUFFER FOR EACH KERNEL EXECUTION ARE NOT RELATED TO THE TOTAL NUMBER
//OF ELEMENTS (ABOVE).
// #define NUMBATCHES 20
// #define BATCHBUFFERELEM 100000000 //THE SMALLER SIZE ALLOCATED ON THE DEVICE FOR EACH KERNEL EXECUTION

// #define GPUSTREAMS 1 //number of concurrent gpu streams, now defined in params.h

using std::cout;
using std::endl;

//sort ascending
bool compareByPointValue(const key_val_sort &a, const key_val_sort &b)
{
    return a.value_at_dim < b.value_at_dim;
}

uint64_t getLinearID_nDimensions2(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    uint64_t index = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	     index += (uint64_t)indexes[i] * multiplier;
  	      multiplier *= dimLen[i];
	}

	return index;
}





////////////////////////////////////////////////////////////////////////////////





void gridIndexingGPU(
    unsigned int * DBSIZE,
    uint64_t totalCells,
    DTYPE * database,
    DTYPE ** dev_database,
    DTYPE * epsilon,
    DTYPE ** dev_epsilon,
    DTYPE * minArr,
    DTYPE ** dev_minArr,
    struct grid ** index,
    struct grid ** dev_index,
    unsigned int * indexLookupArr,
    unsigned int ** dev_indexLookupArr,
    struct gridCellLookup ** gridCellLookupArr,
    struct gridCellLookup ** dev_gridCellLookupArr,
    unsigned int * nNonEmptyCells,
    unsigned int ** dev_nNonEmptyCells,
    unsigned int * nCells,
    unsigned int ** dev_nCells)
{

    cudaError_t errCode;

    double tStartAllocGPU = omp_get_wtime();

    errCode = cudaMalloc( (void**)dev_database, sizeof(DTYPE) * (GPUNUMDIM) * (*DBSIZE));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Alloc database -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMalloc( (void**)dev_epsilon, sizeof(DTYPE));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Alloc epsilon -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMalloc((void**)dev_minArr, sizeof(DTYPE) * (NUMINDEXEDDIM));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Alloc minArr -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMalloc( (void**)dev_indexLookupArr, sizeof(unsigned int) * (*DBSIZE));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: lookup array allocation -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMalloc((void**)dev_nNonEmptyCells, sizeof(unsigned int));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Alloc nNonEmptyCells -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMalloc((void**)dev_nCells, sizeof(unsigned int) * (NUMINDEXEDDIM));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Alloc nCells -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    uint64_t * dev_pointCellArr;
    errCode = cudaMalloc((void**)&dev_pointCellArr, sizeof(uint64_t) * (*DBSIZE));
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: point cell array alloc -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    unsigned int * dev_databaseVal;
	errCode = cudaMalloc((void**)&dev_databaseVal, sizeof(unsigned int) * (*DBSIZE));
	if (errCode != cudaSuccess) {
    	cout << "[INDEX] ~ Error: Alloc databaseVal -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    unsigned int * N = new unsigned int;
	unsigned int * dev_N;
	errCode = cudaMalloc((void**)&dev_N, sizeof(unsigned int) * GPUSTREAMS);
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Alloc dev_N -- error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    double tEndAllocGPU = omp_get_wtime();
    cout << "[INDEX] ~ Time to allocate on the GPU: " << tEndAllocGPU - tStartAllocGPU << "\n\n";
    cout.flush();



    ////////////////////////////////////////////////////////////////////////////



    double tStartCopyGPU = omp_get_wtime();

    errCode = cudaMemcpy( (*dev_database), database, sizeof(DTYPE) * (GPUNUMDIM) * (*DBSIZE), cudaMemcpyHostToDevice );
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: database copy to device -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy( (*dev_epsilon), epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: epsilon copy to device -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy( (*dev_minArr), minArr, sizeof(DTYPE) * (NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Copy minArr to device -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy( (*dev_nCells), nCells, sizeof(unsigned int) * (NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Copy nCells to device -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy(dev_N, DBSIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: database size Got error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    double tEndCopyGPU = omp_get_wtime();
    cout << "[INDEX] ~ Time to copy to the GPU: " << tEndCopyGPU - tStartCopyGPU << "\n\n";
    cout.flush();



    ////////////////////////////////////////////////////////////////////////////



    const int TOTALBLOCKS = ceil((1.0 * (*DBSIZE)) / (1.0 * BLOCKSIZE));
	printf("[INDEX] ~ Total blocks: %d\n",TOTALBLOCKS);

	kernelIndexComputeNonemptyCells<<<TOTALBLOCKS, BLOCKSIZE>>>((*dev_database), dev_N, (*dev_epsilon), (*dev_minArr),
            (*dev_nCells), dev_pointCellArr, nullptr, false);
    cudaDeviceSynchronize();

    thrust::device_ptr<uint64_t> dev_pointCellArr_ptr(dev_pointCellArr);
	thrust::device_ptr<uint64_t> dev_new_end;

	try
    {
		//first sort
		thrust::sort(thrust::device, dev_pointCellArr_ptr, dev_pointCellArr_ptr + (*DBSIZE)); //, thrust::greater<uint64_t>()
		//then unique
		dev_new_end = thrust::unique(thrust::device, dev_pointCellArr_ptr, dev_pointCellArr_ptr + (*DBSIZE));
	}
	catch(std::bad_alloc &e)
	{
	 	std::cerr << "[INDEX] ~ Ran out of memory while sorting" << std::endl;
	    exit(-1);
    }

    uint64_t * new_end = thrust::raw_pointer_cast(dev_new_end);
    uint64_t numNonEmptyCells = std::distance(dev_pointCellArr_ptr, dev_new_end);
    printf("[INDEX] ~ Number of full cells (non-empty): %lu\n", numNonEmptyCells);
    *nNonEmptyCells = numNonEmptyCells;

    (*gridCellLookupArr) = new struct gridCellLookup[numNonEmptyCells];
    uint64_t * pointCellArrTmp = new uint64_t[numNonEmptyCells];
    errCode = cudaMemcpy(pointCellArrTmp, dev_pointCellArr, sizeof(uint64_t) * numNonEmptyCells, cudaMemcpyDeviceToHost);
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: pointCellArrTmp memcpy Got error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

	for (uint64_t i = 0; i < numNonEmptyCells; ++i)
	{
		(*gridCellLookupArr)[i].idx = i;
		(*gridCellLookupArr)[i].gridLinearID = pointCellArrTmp[i];
	}

    kernelIndexComputeNonemptyCells<<<TOTALBLOCKS, BLOCKSIZE>>>((*dev_database), dev_N, (*dev_epsilon), (*dev_minArr),
            (*dev_nCells), dev_pointCellArr, dev_databaseVal, true);

    try
	{
    	thrust::sort_by_key(thrust::device, dev_pointCellArr, dev_pointCellArr + (*DBSIZE), dev_databaseVal);
	}
	catch(std::bad_alloc &e)
	{
		std::cerr << "[INDEX] ~ Ran out of memory while sorting key/value pairs" << std::endl;
	    exit(-1);
	}

    uint64_t * cellKey = new uint64_t[(*DBSIZE)];
    errCode = cudaMemcpy(cellKey, dev_pointCellArr, sizeof(uint64_t) * (*DBSIZE), cudaMemcpyDeviceToHost);
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: pointCellArr memcpy Got error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy(indexLookupArr, dev_databaseVal, sizeof(unsigned int) * (*DBSIZE), cudaMemcpyDeviceToHost);
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: databaseIDValue memcpy Got error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    (*index) = new grid[numNonEmptyCells];
    (*index)[0].indexmin = 0;
	uint64_t cnt=0;
	for (uint64_t i = 1; i < (*DBSIZE); ++i)
    {
		if (cellKey[i - 1] != cellKey[i])
		{
			//grid index
			cnt++;
			(*index)[cnt].indexmin = i;
			(*index)[cnt - 1].indexmax = i - 1;
		}
	}
    delete[] cellKey;
	(*index)[numNonEmptyCells - 1].indexmax = (*DBSIZE) - 1;

    printf("[INDEX] ~ Full cells: %d (%f, fraction full)\n", (unsigned int)numNonEmptyCells, numNonEmptyCells * 1.0 / double(totalCells));
	printf("[INDEX] ~ Empty cells: %ld (%f, fraction empty)\n", totalCells - (unsigned int)numNonEmptyCells, (totalCells - numNonEmptyCells * 1.0) / double(totalCells));
	printf("[INDEX] ~ Size of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f\n",
        (double)sizeof(struct grid) * (totalCells) / (1024.0 * 1024.0 * 1024.0));
	printf("[INDEX] ~ Size of compressed index to be sent to GPU (GiB), excluding the data and grid lookup arr: %f\n",
        (double)sizeof(struct grid) * (numNonEmptyCells * 1.0) / (1024.0 * 1024.0 * 1024.0));
	printf("[INDEX] ~ When copying from entire index to compressed index: number of non-empty cells: %lu\n", numNonEmptyCells);

    ////////////////////////////////////////////////////////////////////////////

    errCode = cudaMalloc( (void**)dev_index, sizeof(struct grid) * (*nNonEmptyCells));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: Alloc grid index -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMalloc( (void**)dev_gridCellLookupArr, sizeof(struct gridCellLookup) * (*nNonEmptyCells));
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: copy grid cell lookup array allocation -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    ////////////////////////////////////////////////////////////////////////////

    errCode = cudaMemcpy( (*dev_nNonEmptyCells), nNonEmptyCells, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if (errCode != cudaSuccess)
    {
		cout << "[INDEX] ~ Error: nNonEmptyCells copy to device -- error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy((*dev_index), (*index), sizeof(struct grid) * numNonEmptyCells, cudaMemcpyHostToDevice);
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: index copy to the GPU error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy((*dev_indexLookupArr), indexLookupArr, sizeof(unsigned int) * (*DBSIZE), cudaMemcpyHostToDevice);
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: index lookup array copy to the GPU error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy((*dev_gridCellLookupArr), (*gridCellLookupArr), sizeof(struct gridCellLookup) * numNonEmptyCells, cudaMemcpyHostToDevice);
	if (errCode != cudaSuccess)
    {
    	cout << "[INDEX] ~ Error: grid lookup array copy to the GPU error with code " << errCode << '\n';
        cout << "[INDEX] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    ////////////////////////////////////////////////////////////////////////////

    delete N;
    delete[] pointCellArrTmp;
    cudaFree(dev_pointCellArr);
    cudaFree(dev_databaseVal);
    cudaFree(dev_N);

    double tEndIndexGPU = omp_get_wtime();
    cout << "[INDEX] ~ Time to index using the GPU (including allocating and transfering memory): " << tEndIndexGPU - tStartAllocGPU << '\n';
    cout.flush();

}




unsigned long long GPUBatchEst_v2(
    int searchMode,
    unsigned int * DBSIZE,
    float staticPartition,
    DTYPE * dev_database,
    unsigned int * dev_originPointIndex,
    DTYPE * dev_epsilon,
    struct grid * dev_grid,
    unsigned int * dev_indexLookupArr,
    struct gridCellLookup * dev_gridCellLookupArr,
    DTYPE * dev_minArr,
    unsigned int * dev_nCells,
    unsigned int * dev_nNonEmptyCells,
    unsigned int * retNumBatches,
    unsigned int * retGPUBufferSize,
    std::vector< std::pair<unsigned int, unsigned int> > * batches)
{

    cudaError_t errCode;

    cout << "[GPU] ~ Estimating batches\n";

    // Parameters for the batch size estimation.
    double sampleRate = 0.10;
    int offsetRate = 1.0 / sampleRate;
    cout << "[GPU] ~ Sample rate: " << sampleRate << ", offset: " << offsetRate << '\n';

    /////////////////
	// N GPU threads
	////////////////

    unsigned int * dev_N_batchEst;
    unsigned int * N_batchEst = new unsigned int;

    unsigned int partitionedDBSIZE = (*DBSIZE) * staticPartition;

    if (SM_HYBRID_STATIC == searchMode && STATIC_SPLIT_QUERIES)
    {
        // Split the worked based on the number of queries, so also reduce the number of queries to estimate
        (*N_batchEst) = partitionedDBSIZE * sampleRate;
    } else {
        // Searchmode is either GPU alone, dynamic hybrid, or the workload is statically split
        //  based on the number of candidate points to refine, and so we estimate all the query points
        //  in all mentionned cases
        (*N_batchEst) = (*DBSIZE) * sampleRate;
    }

    errCode = cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int));
	if (errCode != cudaSuccess)
    {
    	cout << "[GPU] ~ Error: dev_N_batchEst Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy(dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (errCode != cudaSuccess)
    {
	    cout << "[GPU] ~ Error: N batchEST Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    /////////////
	// count the result set size
	////////////

    unsigned int * dev_cnt_batchEst;
    unsigned int * cnt_batchEst = new unsigned int;
    (*cnt_batchEst) = 0;

    errCode = cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int));
	if (errCode != cudaSuccess)
    {
    	cout << "[GPU] ~ Error: dev_cnt_batchEst Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy(dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (errCode != cudaSuccess)
    {
    	cout << "[GPU] ~ Error: dev_cnt_batchEst Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    //////////////////
	// Sample offset - To sample the data to estimate the total number of key/value pairs
	/////////////////

    unsigned int * dev_sampleOffset;
    unsigned int * sampleOffset = new unsigned int;
    (*sampleOffset) = offsetRate;

    errCode = cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int));
	if (errCode != cudaSuccess)
    {
    	cout << "[GPU] ~ Error: sample offset Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy(dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (errCode != cudaSuccess)
    {
    	cout << "[GPU] ~ Error: dev_sampleOffset Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    //////////////////
    // To save the estimated number of neighbors of points
    //////////////////

    unsigned int * dev_estimatedResult;
    unsigned int * estimatedResult = new unsigned int[(*N_batchEst)];

    errCode = cudaMalloc((void**)&dev_estimatedResult, (*N_batchEst) * sizeof(unsigned int));
	if (errCode != cudaSuccess)
    {
    	cout << "[GPU] ~ Error: estimated result Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    int nbBlockTmp;
    if (searchMode == SM_HYBRID_STATIC)
    {
        #if STATIC_SPLIT_QUERIES
            nbBlockTmp = ceil((1.0 * partitionedDBSIZE * sampleRate) / (1.0 * BLOCKSIZE));
        #else
            nbBlockTmp = ceil((1.0 * (*DBSIZE) * sampleRate) / (1.0 * BLOCKSIZE));
        #endif
    } else {
        nbBlockTmp = ceil((1.0 * (*DBSIZE) * sampleRate) / (1.0 * BLOCKSIZE));
    }
    cout << "[GPU] ~ Total blocks: " << nbBlockTmp << '\n';
    cout.flush();

    cout << "[GPU] ~ Estimating batch without using pattern\n";
    cout.flush();

    const int TOTALBLOCKSBATCHEST = nbBlockTmp;


    #if SORT_BY_WORKLOAD
        kernelNDGridIndexBatchEstimator_v2<<<TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_N_batchEst, dev_sampleOffset,
            dev_database, dev_originPointIndex, dev_epsilon, dev_grid, dev_indexLookupArr, dev_gridCellLookupArr, dev_minArr,
            dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells, dev_estimatedResult);
    #else
        kernelNDGridIndexBatchEstimator_v2<<<TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_N_batchEst, dev_sampleOffset,
            dev_database, nullptr, dev_epsilon, dev_grid, dev_indexLookupArr, dev_gridCellLookupArr, dev_minArr,
            dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells, dev_estimatedResult);
    #endif


    cout << "[GPU] ~ ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: " << cudaGetLastError() << '\n';
    cout.flush();

    errCode = cudaMemcpy(cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (errCode != cudaSuccess)
    {
	    cout << "[GPU] ~ Error: getting cnt for batch estimate from GPU Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	} else {
        cout << "[GPU] ~ Result set size for estimating the number of batches (sampled): " << *cnt_batchEst << '\n';
        cout.flush();
	}

    errCode = cudaMemcpy(estimatedResult, dev_estimatedResult, (*N_batchEst) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (errCode != cudaSuccess)
    {
	    cout << "[GPU] ~ Error: getting estimated results for batch estimate from GPU Got error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

    unsigned int GPUBufferSize = 50000000;
    // unsigned int GPUBufferSize = 100000000;

    // uint64_t estimatedNeighbors = (uint64_t)*cnt_batchEst * (uint64_t)offsetRate;
    // cout << "[GPU] ~ From GPU cnt: " << *cnt_batchEst <<", offset rate: " << offsetRate << '\n';
    // cout.flush();

    unsigned long long fullEst = 0;
    unsigned int * estimatedFull;

    unsigned int nbUnestimatedSequences;

    if (SM_HYBRID_STATIC == searchMode)
    {
        #if STATIC_SPLIT_QUERIES
            nbUnestimatedSequences = partitionedDBSIZE / (*sampleOffset);
            estimatedFull = new unsigned int[partitionedDBSIZE];
        #else
            nbUnestimatedSequences = (*DBSIZE) / (*sampleOffset);
            estimatedFull = new unsigned int[(*DBSIZE)];
        #endif
    } else {
        nbUnestimatedSequences = (*DBSIZE) / (*sampleOffset);
        estimatedFull = new unsigned int[(*DBSIZE)];
    }

    for (int i = 0; i < nbUnestimatedSequences - 1; ++i)
    {
        unsigned int nbEstBefore = estimatedResult[i];
        unsigned int nbEstAfter = estimatedResult[i + 1];
        unsigned int maxEst = (nbEstBefore < nbEstAfter) ? nbEstAfter : nbEstBefore;

        unsigned int estBefore = i * (*sampleOffset);
        unsigned int estAfter = (i + 1) * (*sampleOffset);
        estimatedFull[estBefore] = nbEstBefore;
        fullEst += nbEstBefore;

        for (int j = estBefore + 1; j < estAfter; ++j)
        {
            #if SORT_BY_WORKLOAD
                estimatedFull[j] = maxEst;
                fullEst += maxEst;
            #else
                // If we do not sort by workload, then we can not assume that the work is in non-increasing order,
                // and thus that the used estimator is "correct", so we overestimate the estimation to compensate,
                // similarly as in the original algorithm
                estimatedFull[j] = maxEst + maxEst * sampleRate;
                fullEst += maxEst + maxEst * sampleRate;
            #endif
        }

    }

    cout << "[GPU | RESULT] ~ Total estimated workload: " << fullEst << '\n';

    if (searchMode == SM_HYBRID_STATIC)
    {
        // Not enough work to fill at least GPUSTREAMS batches, so reducing GPUBufferSize so the
        // GPU can fully use its GPUSTREAMS streams
        // Used if the work is statically partitioned, as the CPU will always have some work reserved
        if (fullEst < (GPUBufferSize * GPUSTREAMS))
        {
            GPUBufferSize = fullEst / (GPUSTREAMS);
            cout << "[GPU] ~ Too few batches, reducing GPUBufferSize to " << GPUBufferSize << '\n';
        }
    } else {
        // Not enough work to fill at least 6 batches (2 * GPUSTREAMS)
        // So we force to have at least 6 batches so all streams can be used, and the CPU as well
        // Used if the work is dynamically partitioned (work queue), so the CPU can have some work
        if (fullEst < (GPUBufferSize * GPUSTREAMS * 2))
        {
            GPUBufferSize = fullEst / (GPUSTREAMS * 2);
            cout << "[GPU] ~ Too few batches, reducing GPUBufferSize to " << GPUBufferSize << '\n';
        }
    }

    unsigned int batchBegin = 0;
    unsigned int batchEnd = 0;
    unsigned long long runningEst = 0;
    // Keeping 5% of margin to avoid a potential overflow of the buffer
    unsigned int reserveBuffer = GPUBufferSize * 0.05;

    if (searchMode == SM_HYBRID_STATIC)
    {
        #if STATIC_SPLIT_QUERIES
            for (int i = 0; i < partitionedDBSIZE; ++i)
            {
                runningEst += estimatedFull[i];
                // fullEst += estimatedFull[i];
                if ((GPUBufferSize - reserveBuffer) <= runningEst)
                {
                    batchEnd = i;
                    batches->push_back(std::make_pair(batchBegin, batchEnd));
                    batchBegin = i;
                    runningEst = 0;
                } else {
                    // The last batch may not fulfill the above condition of filling a result buffer
                    if (partitionedDBSIZE - 1 == i)
                    {
                        batchEnd = partitionedDBSIZE;
                        batches->push_back(std::make_pair(batchBegin, batchEnd));
                    }
                }
            }
            printf("[GPU | RESULT] ~ %u query points allocated to the GPU, with %llu estimated candidates\n", partitionedDBSIZE, runningEst);
            printf("[GPU | RESULT] ~ %u query points allocated to the CPU, with %llu estimated candidates\n", (*DBSIZE) - partitionedDBSIZE, fullEst - runningEst);
            setQueueIndex(partitionedDBSIZE);
        #else // Static partitioning based on the number candidate points to refine
            // unsigned long long partitionedCandidates = fullEst * staticPartition;
            // runningEst = 0;
            // unsigned long long runningEstBatch = 0;
            // unsigned int queryPoint = 0;
            // while (runningEst < partitionedCandidates)
            // {
            //     runningEst += estimatedFull[queryPoint];
            //     runningEstBatch += estimatedFull[queryPoint];
            //     if ((GPUBufferSize - reserveBuffer) <= runningEstBatch)
            //     {
            //         batchEnd = queryPoint;
            //         batches->push_back(std::make_pair(batchBegin, batchEnd));
            //         batchBegin = queryPoint;
            //         runningEstBatch = 0;
            //     }
            //     queryPoint++;
            // }
            // batchEnd = queryPoint;
            // batches->push_back(std::make_pair(batchBegin, batchEnd));
            for (int i = 0; i < (*DBSIZE); ++i)
            {
                runningEst += estimatedFull[i];
                // fullEst += estimatedFull[i];
                if ((GPUBufferSize - reserveBuffer) <= runningEst)
                {
                    batchEnd = i;
                    batches->push_back(std::make_pair(batchBegin, batchEnd));
                    batchBegin = i;
                    runningEst = 0;
                } else {
                    // The last batch may not fulfill the above condition of filling a result buffer
                    if ((*DBSIZE) - 1 == i)
                    {
                        batchEnd = (*DBSIZE);
                        batches->push_back(std::make_pair(batchBegin, batchEnd));
                    }
                }
            }

            // printf("[GPU | RESULT] ~ %u query points allocated to the GPU, with %llu estimated candidates\n", queryPoint, runningEst);
            // printf("[GPU | RESULT] ~ %u query points allocated to the CPU, with %llu estimated candidates\n", (*DBSIZE) - queryPoint, fullEst - runningEst);
            setQueueIndex((*DBSIZE));
        #endif
        fullEst = runningEst;
    } else {
        for (int i = 0; i < (*DBSIZE); ++i)
        {
            runningEst += estimatedFull[i];
            // fullEst += estimatedFull[i];
            if ((GPUBufferSize - reserveBuffer) <= runningEst)
            {
                batchEnd = i;
                batches->push_back(std::make_pair(batchBegin, batchEnd));
                batchBegin = i;
                runningEst = 0;
            } else {
                // The last batch may not fulfill the above condition of filling a result buffer
                if ((*DBSIZE) - 1 == i)
                {
                    batchEnd = (*DBSIZE);
                    batches->push_back(std::make_pair(batchBegin, batchEnd));
                }
            }
        }
        // setQueueIndex((batches[GPUSTREAMS]).first);
    }

    cout << "[GPU] ~ Estimated total result set size: " << fullEst << '\n';
    cout << "[GPU] ~ Number of batches: " << batches->size() << '\n';
    cout.flush();

    (*retNumBatches) = batches->size();
    (*retGPUBufferSize) = GPUBufferSize;

    cout << "[GPU] ~ Done estimating batches\n";

    cudaFree(dev_cnt_batchEst);
    cudaFree(dev_N_batchEst);
    cudaFree(dev_sampleOffset);
    cudaFree(dev_estimatedResult);

    delete[] estimatedResult;
    delete[] estimatedFull;
    delete N_batchEst;
    delete cnt_batchEst;
    delete sampleOffset;

    return fullEst;

}






//modified from: makeDistanceTableGPUGridIndexBatchesAlternateTest
void distanceTableNDGridBatches(
        int searchMode,
        float staticPartition,
        unsigned int * DBSIZE,
        DTYPE * epsilon,
        DTYPE * dev_epsilon,
        DTYPE * database,
        DTYPE * dev_database,
        struct grid * grid,
        struct grid * dev_grid,
        unsigned int * indexLookupArr,
        unsigned int * dev_indexLookupArr,
        struct gridCellLookup * gridCellLookupArr,
        struct gridCellLookup * dev_gridCellLookupArr,
        DTYPE * minArr,
        DTYPE * dev_minArr,
        unsigned int * nCells,
        unsigned int * dev_nCells,
        unsigned int * nNonEmptyCells,
        unsigned int * dev_nNonEmptyCells,
        // unsigned int * gridCellNDMask,
        // unsigned int * dev_gridCellNDMask,
        // unsigned int * gridCellNDMaskOffsets,
        // unsigned int * dev_gridCellNDMaskOffsets,
        // unsigned int * nNDMaskElems,
        unsigned int * originPointIndex,
        unsigned int * dev_originPointIndex,
        struct neighborTableLookup * neighborTable,
        std::vector<struct neighborDataPtrs> * pointersToNeighbors,
        uint64_t * totalNeighbors,
        unsigned int * nbQueriesGPU)
{
    double tKernelResultsStart = omp_get_wtime();

  	//CUDA error code:
  	cudaError_t errCode;

  	cout << "\n[GPU] ~ Sometimes the GPU will error on a previous execution and you won't know. \n[GPU] ~ Last error start of function: " << cudaGetLastError() << '\n';
    cout.flush();



  	///////////////////////////////////
  	//COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
  	///////////////////////////////////

  	//total size of the result set as it's batched
  	//this isnt sent to the GPU
  	unsigned int * totalResultSetCnt = new unsigned int;
  	*totalResultSetCnt = 0;

  	//count values - for an individual kernel launch
  	//need different count values for each stream
  	unsigned int * cnt;
  	cnt = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);
  	*cnt = 0;

  	unsigned int * dev_cnt;
  	dev_cnt = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);
  	*dev_cnt = 0;

    for (int i = 0; i < (GPUSTREAMS); i++) {
        cnt[i] = 0;
        dev_cnt[i] = 0;
    }

  	//allocate on the device
  	errCode = cudaMalloc((void**)&dev_cnt, sizeof(unsigned int) * GPUSTREAMS);
  	if (errCode != cudaSuccess)
    {
		cout << "[GPU] ~ Error: Alloc cnt -- error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
  	}

  	///////////////////////////////////
  	//END COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
  	///////////////////////////////////



  	////////////////////////////////////
  	//NUMBER OF THREADS PER GPU STREAM
  	////////////////////////////////////

  	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
  	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
  	unsigned int * N = new unsigned int[GPUSTREAMS];

  	unsigned int * dev_N;
  	// dev_N = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);

  	//allocate on the device
  	errCode = cudaMalloc((void**)&dev_N, sizeof(unsigned int) * GPUSTREAMS);
  	if (errCode != cudaSuccess)
    {
		cout << "[GPU] ~ Error: Alloc dev_N -- error with code " << errCode << '\n';
        cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
	}

  	////////////////////////////////////
  	//NUMBER OF THREADS PER GPU STREAM
  	////////////////////////////////////



    /////////////////////////////////////////////////////////
  	//BEGIN BATCH ESTIMATOR
  	/////////////////////////////////////////////////////////

  	unsigned long long estimatedNeighbors = 0;
  	unsigned int numBatches = 0;
  	unsigned int GPUBufferSize = 0;

    std::vector< std::pair<unsigned int, unsigned int> > batchesVector;

	double tstartbatchest = omp_get_wtime();
    if (SM_HYBRID_STATIC == searchMode)
    {
        #if STATIC_SPLIT_QUERIES
            #if SORT_BY_WORKLOAD
                estimatedNeighbors = GPUBatchEst_v2(searchMode, DBSIZE, staticPartition, dev_database, dev_originPointIndex, dev_epsilon, dev_grid, dev_indexLookupArr,
                        dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize, &batchesVector);
            #else
                estimatedNeighbors = GPUBatchEst_v2(searchMode, DBSIZE, staticPartition, dev_database, nullptr, dev_epsilon, dev_grid, dev_indexLookupArr,
                        dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize, &batchesVector);
            #endif
        #else
            unsigned int nbQueryPointsStatic = getStaticQueryPoint();
            cout << "[GPU | DEBUG] ~ Number of queries for the GPU: " << nbQueryPointsStatic << '\n';
            #if SORT_BY_WORKLOAD
                estimatedNeighbors = GPUBatchEst_v2(searchMode, &nbQueryPointsStatic, staticPartition, dev_database, dev_originPointIndex, dev_epsilon, dev_grid, dev_indexLookupArr,
                        dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize, &batchesVector);
            #else
                estimatedNeighbors = GPUBatchEst_v2(searchMode, &nbQueryPointsStatic, staticPartition, dev_database, nullptr, dev_epsilon, dev_grid, dev_indexLookupArr,
                        dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize, &batchesVector);
            #endif
        #endif
    } else {
        #if SORT_BY_WORKLOAD
            estimatedNeighbors = GPUBatchEst_v2(searchMode, DBSIZE, staticPartition, dev_database, dev_originPointIndex, dev_epsilon, dev_grid, dev_indexLookupArr,
                    dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize, &batchesVector);
        #else
            estimatedNeighbors = GPUBatchEst_v2(searchMode, DBSIZE, staticPartition, dev_database, nullptr, dev_epsilon, dev_grid, dev_indexLookupArr,
                    dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize, &batchesVector);
        #endif
    }
    double tendbatchest = omp_get_wtime();

    cout << "[GPU] ~ Time to estimate batches: " << tendbatchest - tstartbatchest << '\n';
    cout.flush();

    cout << "[GPU] ~ In calling function: Estimated neighbors = " << estimatedNeighbors
         << ", num. batches = " << numBatches << ", GPU buffer size = " << GPUBufferSize << '\n';
    cout.flush();

    // cout << "[GPU] ~ Batches: \n";
    // for (int i = 0; i < batchesVector.size(); ++i)
    // {
    //     cout << "   [GPU] ~ " << batchesVector[i].first << ", " << batchesVector[i].second <<  '\n';
    // }

    // sets the batch size for the queue and the queue index, considering the offset reserved for the GPU
    // shouldn't happen anymore as we always have at least 2*GPUSTREAMS batches now
    // setQueueIndex(GPUSTREAMS * (*DBSIZE / numBatches));
    // if (batchesVector.size() < GPUSTREAMS)
    // {
    //     setQueueIndex((*DBSIZE)); // the GPU reserves all the computation
    // } else {
    if (searchMode != SM_HYBRID_STATIC)
    {
        setQueueIndex(batchesVector[GPUSTREAMS].first);
    }
      // }

    // setQueueIndex(0);

  	/////////////////////////////////////////////////////////
  	//END BATCH ESTIMATOR
  	/////////////////////////////////////////////////////////



  	///////////////////
  	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
  	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
    unsigned int datasetSize = *DBSIZE;
    size_t freeMem = 0;
    size_t totalMem = 0;
    int memGetSuccess = cudaMemGetInfo(&freeMem, &totalMem);
    if (memGetSuccess != cudaSuccess) {
        cout << "You probably don't have a gpu, in which case, how did you get this far?" << endl;
        exit(memGetSuccess);
    }
    #if !SILENT_GPU
        cout << "[GPU] ~ Total Memory: " << totalMem << '\n';
        cout.flush();
    #endif

    // Compute Total blocks computable in a single instances based on estimated size
    // Have to convert esitmatedNeighbors into bytes
    const unsigned int tpp = 32;
    cout << "DEFINED TPP AS " << tpp << endl;

    numBatches = ceil(1.0 * (estimatedNeighbors * 2 * sizeof(int)) / (0.9 * GPUBufferSize));
    uint64_t pointsPerBlock = BLOCKSIZE / tpp;
    uint64_t NBLOCKS = ceil(((1.0 * datasetSize) / (1.0 * numBatches)) / (1.0 * pointsPerBlock));
    // NBLOCKS = NBLOCKS + (NBLOCKS % BLOCKSIZE);
    // numBatches = datasetSize / NBLOCKS;
    cout << "[GPU] ~ Number of blocks for a single instance: " << NBLOCKS << endl;
    cout << "[GPU] ~ Total Iterations: " << numBatches << '\n';
    cout.flush();
    // #if !SILENT_GPU
    //     cout << "[GPU] ~ Number of blocks for a single instance: " << NBLOCKS << endl;
    //     cout << "[GPU] ~ Total Iterations: " << numBatches << '\n';
    //     cout.flush();
    // #endif
	for (int i = 0; i < numBatches; i++)
    {
		int *ptr;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr = ptr;
		tmpStruct.sizeOfDataArr = 0;

		pointersToNeighbors->push_back(tmpStruct);
	}

  	///////////////////
  	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
  	///////////////////



  	///////////////////////////////////
  	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
  	///////////////////////////////////

  	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS
  	//GPU BUFFER ON THE DEVICE
  	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
  	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
  	//EXECUTING STREAMS ON THE HOST

	//GPU MEMORY ALLOCATION: key/value pairs

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value
	for (int i = 0; i < GPUSTREAMS; i++)
	{
		errCode = cudaMalloc((void **)&dev_pointIDKey[i], 2 * sizeof(int) * GPUBufferSize);
		if (errCode != cudaSuccess)
        {
			cout << "[GPU] ~ CUDA: Got error with code " << errCode << '\n'; //2 means not enough memory
            cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
            cout.flush();
		}

		errCode = cudaMalloc((void **)&dev_pointInDistValue[i], 2 * sizeof(int) * GPUBufferSize);
		if (errCode != cudaSuccess)
        {
			cout << "[GPU] ~ CUDA: Got error with code " << errCode << '\n'; //2 means not enough memory
            cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
            cout.flush();
		}

	}
    cout << "[GPU] ~ Allocation pointIDKey and pointInDistValue on the GPU, size = " << 2 * sizeof(int) * GPUBufferSize << '\n';
    cout.flush();

  	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
  	//ON THE HOST
  	//pinned result set memory for the host
  	//the number of elements are recorded for that batch in resultElemCountPerBatch
  	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS
  	//can't do async copies without pinned memory

  	//PINNED MEMORY TO COPY FROM THE GPU
  	int * pointIDKey[GPUSTREAMS]; //key
  	int * pointInDistValue[GPUSTREAMS]; //value

  	double tstartpinnedresults = omp_get_wtime();

    #pragma omp parallel for num_threads(GPUSTREAMS)
  	for (int i = 0; i < GPUSTREAMS; i++)
  	{
  		  cudaMallocHost((void **) &pointIDKey[i], 2 * sizeof(int) * GPUBufferSize);
  		  cudaMallocHost((void **) &pointInDistValue[i], 2 * sizeof(int) * GPUBufferSize);
  	}

  	double tendpinnedresults = omp_get_wtime();

    cout << "[GPU] ~ Time to allocate pinned memory for results: " << tendpinnedresults - tstartpinnedresults << '\n';
    cout.flush();

  	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
  	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);

    cout << "[GPU] ~ Memory request for results on GPU (GiB): " << (double)(sizeof(int) * 2 * GPUBufferSize * GPUSTREAMS) / (1024 * 1024 * 1024) << '\n';
    cout.flush();
    cout << "[GPU] ~ Memory requested for results in MAIN MEMORY (GiB): " << (double)(sizeof(int) * 2 * GPUBufferSize * GPUSTREAMS) / (1024 * 1024 * 1024) << '\n';
    cout.flush();

  	///////////////////////////////////
  	//END ALLOCATE MEMORY FOR THE RESULT SET
  	///////////////////////////////////



  	/////////////////////////////////
  	//SET OPENMP ENVIRONMENT VARIABLES
  	////////////////////////////////

  	omp_set_num_threads(GPUSTREAMS);

  	/////////////////////////////////
  	//END SET OPENMP ENVIRONMENT VARIABLES
  	////////////////////////////////



  	/////////////////////////////////
  	//CREATE STREAMS
  	////////////////////////////////

  	cudaStream_t stream[GPUSTREAMS];

  	for (int i = 0; i < GPUSTREAMS; i++)
    {
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
  	}



  	///////////////////////////////////
  	//LAUNCH KERNEL IN BATCHES
  	///////////////////////////////////

  	//since we use the strided scheme, some of the batch sizes
  	//are off by 1 of each other, a first group of batches will
  	//have 1 extra data point to process, and we calculate which batch numbers will
  	//have that.  The batchSize is the lower value (+1 is added to the first ones)


    // unsigned int batchSize = (*DBSIZE) / numBatches;
    unsigned int batchSize = datasetSize / numBatches;
    // unsigned int batchesThatHaveOneMore = (*DBSIZE) - (batchSize * numBatches); //batch number 0- < this value have one more
    unsigned int batchesThatHaveOneMore = datasetSize - (batchSize * numBatches);
    cout << "[GPU] ~ Batches that have one more GPU thread: " << batchesThatHaveOneMore << " batchSize(N): " << batchSize << '\n';
    cout.flush();

    uint64_t totalResultsLoop = 0;

    unsigned int * batchBegin = new unsigned int[GPUSTREAMS];
    for (int i = 0; i < GPUSTREAMS; i++)
    {
        batchBegin[i] = 0;
    }

    unsigned int * dev_batchBegin;
    errCode = cudaMalloc( (void**)&dev_batchBegin, GPUSTREAMS * sizeof(unsigned int));
    if (errCode != cudaSuccess)
    {
        cout << "[GPU] ~ Error: Alloc queue index -- error with code " << errCode << '\n';
        cout.flush();
    }

    cudaEvent_t * startKernel = new cudaEvent_t[GPUSTREAMS];
    cudaEvent_t * stopKernel = new cudaEvent_t[GPUSTREAMS];
    float * kernelTimes = new float[GPUSTREAMS];
    unsigned int * nbKernelInvocation = new unsigned int[GPUSTREAMS];
    unsigned int * nbQueryPoint = new unsigned int [GPUSTREAMS];
    double computeTime = 0;

    for (int i = 0; i < GPUSTREAMS; ++i)
    {
        cudaEventCreate(&startKernel[i]);
        cudaEventCreate(&stopKernel[i]);
        kernelTimes[i] = 0;
        nbKernelInvocation[i] = 0;
        nbQueryPoint[i] = 0;
    }

    #if GPU_LOCKING
    pthread_mutex_t gpu_lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_init(&gpu_lock, NULL);
    #endif

    if (SM_HYBRID == searchMode)
    {
        unsigned int globalBatchCounter = GPUSTREAMS;

        double tStartCompute = omp_get_wtime();
        #pragma omp parallel reduction(+: totalResultsLoop) num_threads(GPUSTREAMS)
        {
            unsigned int tid = omp_get_thread_num();
            // std::pair<unsigned int, unsigned int> gpuBatch = std::make_pair(tid * batchSize, tid * batchSize + batchSize);
            std::pair<unsigned int, unsigned int> gpuBatch = batchesVector[tid];

            unsigned int localBatchCounter = tid;

            do
            {
                nbQueryPoint[tid] += gpuBatch.second - gpuBatch.first;
                #if !SILENT_GPU
                    printf("[GPU | T_%d] ~ New batch: begin = %d, end = %d\n", tid, gpuBatch.first, gpuBatch.second);
                #endif

                errCode = cudaMemcpy( &dev_batchBegin[tid], &gpuBatch.first, sizeof(unsigned int), cudaMemcpyHostToDevice );
            	if (errCode != cudaSuccess)
                {
            		cout << "[GPU] ~ Error: queue index copy to device -- error with code " << errCode << '\n';
                    cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                    cout.flush();
            	}

                // N[tid] = batchSize;
                N[tid] = gpuBatch.second - gpuBatch.first;
                errCode = cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
        		if (errCode != cudaSuccess)
                {
        			cout << "[GPU] ~ Error: N Got error with code " << errCode << '\n';
                    cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                    cout.flush();
        		}

                // the batched result set size (reset to 0):
        		cnt[tid] = 0;
        		errCode = cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
        		if (errCode != cudaSuccess)
                {
        			cout << "[GPU] ~ Error: dev_cnt memcpy Got error with code " << errCode << '\n';
                    cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                    cout.flush();
        		}

                const int TOTALBLOCKS = ceil( (1.0 * (N[tid])) / (1.0 * BLOCKSIZE) );
                #if !SILENT_GPU
                    cout << "[GPU] ~ Total blocks: " << TOTALBLOCKS << '\n';
                    cout.flush();
                #endif

                #if GPU_LOCKING
                pthread_mutex_lock(&gpu_lock);
                #endif
                // double beginKernel = omp_get_wtime();
                cudaEventRecord(startKernel[tid], stream[tid]);
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobal<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_batchBegin[tid], &dev_N[tid],
                        dev_database, nullptr, dev_originPointIndex, dev_epsilon, dev_grid,
                        dev_indexLookupArr, dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells,
                        dev_pointIDKey[tid], dev_pointInDistValue[tid], tpp);
                #else
                    kernelNDGridIndexGlobal<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_batchBegin[tid], &dev_N[tid],
                        dev_database, nullptr, nullptr, dev_epsilon, dev_grid,
                        dev_indexLookupArr, dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells,
                        dev_pointIDKey[tid], dev_pointInDistValue[tid], tpp);
                #endif
                cudaEventRecord(stopKernel[tid], stream[tid]);


                errCode = cudaGetLastError();
                #if !SILENT_GPU
            		cout << "\n\n[GPU] ~ KERNEL LAUNCH RETURN: " << errCode << '\n';
                    cout.flush();
                #endif
        		if ( cudaSuccess != cudaGetLastError() )
                {
        			cout << "\n\n[GPU] ~ ERROR IN KERNEL LAUNCH. ERROR: " << cudaSuccess << '\n';
                    cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                    cout.flush();
        		}

                // find the size of the number of results
        		errCode = cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
        		if (errCode != cudaSuccess)
                {
        			cout << "[GPU] ~ Error: getting cnt from GPU Got error with code " << errCode << '\n';
                    cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                    cout.flush();
        		}
                #if !SILENT_GPU
        		else{
                    cout << "[GPU] ~ Result set size within epsilon: " << cnt[tid] << '\n';
                    cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                    cout.flush();
        		}
                #endif

        		// double endKernel = omp_get_wtime();

                nbKernelInvocation[tid]++;

                cudaEventSynchronize(stopKernel[tid]);
                float timeKernel = 0;
                cudaEventElapsedTime(&timeKernel, startKernel[tid], stopKernel[tid]);
                kernelTimes[tid] += timeKernel;

                thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
        		thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

                try{
        			thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);
        		}
        		catch(std::bad_alloc &e)
        		{
                    #if GPU_LOCKING
                    pthread_mutex_unlock(&gpu_lock);
                    #endif
        			std::cerr << "[GPU] ~ Ran out of memory while sorting, " << GPUBufferSize << '\n';
                    cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                    cout.flush();
        			exit(1);
        		}
                #if GPU_LOCKING
                pthread_mutex_unlock(&gpu_lock);
                #endif

                cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
            		cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);

                // cudaStreamSynchronize(stream[tid]);

                double tableconstuctstart = omp_get_wtime();
        		//set the number of neighbors in the pointer struct:
        		(*pointersToNeighbors)[localBatchCounter].sizeOfDataArr = cnt[tid];
        		(*pointersToNeighbors)[localBatchCounter].dataPtr = new int[cnt[tid]];

                constructNeighborTableKeyValueWithPtrs(pointIDKey[tid], pointInDistValue[tid], neighborTable, (*pointersToNeighbors)[localBatchCounter].dataPtr, &cnt[tid]);

                double tableconstuctend = omp_get_wtime();

                #if !SILENT_GPU
                    cout << "[GPU] ~ Table construct time: " << tableconstuctend - tableconstuctstart << '\n';
                    cout.flush();
                #endif

                // add the batched result set size to the total count
        		totalResultsLoop += cnt[tid];

                #if !SILENT_GPU
                    cout << "[GPU] ~ Running total of total size of result array, tid: " << tid << ", " << totalResultsLoop << '\n';
                    cout.flush();
                #endif

                // gpuBatch = getBatchFromQueue(*DBSIZE, batchSize);
                gpuBatch = getBatchFromQueue_v2(batchesVector);
                // gpuBatch = getBatchFromQueue(9 * batchSize, batchSize);

                #pragma omp critical
                {
                    localBatchCounter = globalBatchCounter;
                    globalBatchCounter++;
                }

            } while(0 != gpuBatch.second);

        } // parallel section
        double tEndCompute = omp_get_wtime();
        computeTime = tEndCompute - tStartCompute;
    }
    else
    { // searchModes that have a fixed number of queries (e.g., original GPU kernel or static partitioning)
        // errCode = cudaMemcpy( &dev_batchBegin[0], batchBegin, sizeof(unsigned int), cudaMemcpyHostToDevice );
        // if (errCode != cudaSuccess)
        // {
        //     cout << "[GPU] ~ Error: queue index copy to device -- error with code " << errCode << '\n';
        //     cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
        //     cout.flush();
        // }

        double computeTimeStart = omp_get_wtime();

        double * computeTimeArray = new double[GPUSTREAMS];
        for(int i = 0; i < GPUSTREAMS; ++i)
        {
            computeTimeArray[i] = 0;
        }

        //FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
    	//i=0...numBatches
        #pragma omp parallel for schedule(dynamic, 1) reduction(+: totalResultsLoop) num_threads(GPUSTREAMS)
    	for (int i = 0; i < 1; i++)
        // for (int i = 0; i < 9; ++i)
    	{
            int tid = omp_get_thread_num();

            double tStartLoop = omp_get_wtime();

            #if !SILENT_GPU
                cout << "[GPU] ~ tid " << tid << ", starting iteration " << i << " with block: " << (i * NBLOCKS) <<'\n';
                cout.flush();
            #endif

    		//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
    		//AS ONE GPU THREAD PROCESSES A SINGLE POINT
            // int blockResultSize = NBLOCKS;
            batchBegin[tid] = (i * NBLOCKS);
            errCode = cudaMemcpy( &dev_batchBegin[tid], &batchBegin[tid], sizeof(unsigned int), cudaMemcpyHostToDevice );
            if (errCode != cudaSuccess)
            {
                cout << "[GPU] ~ Error: queue index copy to device -- error with code " << errCode << '\n';
                cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                cout.flush();
            }

            N[tid] = NBLOCKS;
            if (((i + 1) * NBLOCKS) > datasetSize) {
                N[tid] = datasetSize - (i * NBLOCKS);
            }
            #if !SILENT_GPU
                cout << "[GPU] ~ N (1 less): " << N[tid] << ", tid " << tid << '\n';
                cout.flush();
            #endif

            nbQueryPoint[tid] += N[tid];

    		//set relevant parameters for the batched execution that get reset

    		//copy N to device
    		//N IS THE NUMBER OF THREADS
    		errCode = cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
    		if (errCode != cudaSuccess)
            {
    			cout << "[GPU] ~ Error: N Got error with code " << errCode << '\n';
                cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                cout.flush();
    		}

    		//the batched result set size (reset to 0):
    		cnt[tid] = 0;
    		errCode = cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
    		if (errCode != cudaSuccess)
            {
    			cout << "[GPU] ~ Error: dev_cnt memcpy Got error with code " << errCode << '\n';
                cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                cout.flush();
    		}

    		// const int TOTALBLOCKS = ceil( (1.0 * (N[tid])) / (1.0 * BLOCKSIZE) );
            #if !SILENT_GPU
                cout << "[GPU] ~ Total blocks: " << NBLOCKS << '\n';
                cout.flush();
            #endif

            #if GPU_LOCKING
            pthread_mutex_lock(&gpu_lock);
            #endif

    		//execute kernel
    		//0 is shared memory pool
            cudaEventRecord(startKernel[tid], stream[tid]);
            #if SORT_BY_WORKLOAD
                kernelNDGridIndexGlobal<<< NBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_batchBegin[tid], &dev_N[tid], 
                    dev_database, nullptr, dev_originPointIndex, dev_epsilon, dev_grid,
                    dev_indexLookupArr, dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells,
                    dev_pointIDKey[tid], dev_pointInDistValue[tid], tpp);
            #else
                kernelNDGridIndexGlobal<<< NBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(&dev_batchBegin[tid], &dev_N[tid],
                    dev_database, nullptr, nullptr, dev_epsilon, dev_grid,
                    dev_indexLookupArr, dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells,
                    dev_pointIDKey[tid], dev_pointInDistValue[tid], tpp);
            #endif
            cudaEventRecord(stopKernel[tid], stream[tid]);


            errCode = cudaGetLastError();
            #if !SILENT_GPU
        		cout << "\n\n[GPU] ~ KERNEL LAUNCH RETURN: " << errCode << '\n';
                cout.flush();
            #endif
    		if ( cudaSuccess != cudaGetLastError() )
            {
    			cout << "\n\n[GPU] ~ ERROR IN KERNEL LAUNCH. ERROR: " << cudaSuccess << '\n';
                cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                cout.flush();
    		}

    		// find the size of the number of results
    		errCode = cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
    		if (errCode != cudaSuccess)
            {
    			cout << "[GPU] ~ Error: getting cnt from GPU Got error with code " << errCode << '\n';
                cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                cout.flush();
    		}
            #if !SILENT_GPU
    		else {
                cout << "[GPU] ~ Result set size within epsilon: " << cnt[tid] << '\n';
                cout << "  Details: " << cudaGetErrorString(errCode) << '\n';
                cout.flush();
    		}
            #endif

    		// double endKernel = omp_get_wtime();
            // kernelTime[tid] += endKernel - beginKernel;
    		//cout << "Single kernel execution time = " << endKernel - beginKernel << " ms" << endl;

            nbKernelInvocation[tid]++;

            cudaEventSynchronize(stopKernel[tid]);
            float timeKernel = 0;
            cudaEventElapsedTime(&timeKernel, startKernel[tid], stopKernel[tid]);
            kernelTimes[tid] += timeKernel;


    		////////////////////////////////////
    		//SORT THE TABLE DATA ON THE GPU
    		//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
    		////////////////////////////////////

    		/////////////////////////////
    		//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
    		//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
    		//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
    		/////////////////////////////

    		//sort by key with the data already on the device:
    		//wrap raw pointer with a device_ptr to use with Thrust functions
    		thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
    		thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

    		//XXXXXXXXXXXXXXXX
    		//THRUST USING STREAMS REQUIRES THRUST V1.8
    		//XXXXXXXXXXXXXXXX

    		try {
    			thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);

    		} catch(std::bad_alloc &e) {
                #if GPU_LOCKING
                pthread_mutex_unlock(&gpu_lock);
                #endif
    			std::cerr << "[GPU] ~ Ran out of memory while sorting, " << GPUBufferSize << '\n';
                cout.flush();
    			exit(1);
    		}

            #if GPU_LOCKING
            pthread_mutex_unlock(&gpu_lock);
            #endif
            // cout << "[GPU] ~ Thrust sort by key\n";
            // cout.flush();
            // cout << "[GPU] ~ Copy size: " << cnt[tid] * sizeof(int) << '\n';
            // cout.flush();

    		// thrust with streams into individual buffers for each batch
    		cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
    		cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);

            // cout << "[GPU] ~ Async memcpy of pointers\n";
            // cout.flush();

    		//need to make sure the data is copied before constructing portion of the neighbor table
    		// cudaStreamSynchronize(stream[tid]);

            // cout << "[GPU] ~ Stream synchronization\n";
            // cout.flush();

    		double tableconstuctstart = omp_get_wtime();
    		//set the number of neighbors in the pointer struct:
    		(*pointersToNeighbors)[i].sizeOfDataArr = cnt[tid];
    		(*pointersToNeighbors)[i].dataPtr = new int[cnt[tid]];

    		constructNeighborTableKeyValueWithPtrs(pointIDKey[tid], pointInDistValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, &cnt[tid]);
    		// cout <<"In make neighbortable. Data array ptr: "<<(*pointersToNeighbors)[i].dataPtr<<" , size of data array: "<<(*pointersToNeighbors)[i].sizeOfDataArr;cout.flush();

    		double tableconstuctend = omp_get_wtime();

            #if !SILENT_GPU
                cout << "[GPU] ~ Table construct time: " << tableconstuctend - tableconstuctstart << '\n';
                cout.flush();
            #endif

    		//add the batched result set size to the total count
            totalResultsLoop+=cnt[tid];

            #if !SILENT_GPU
                cout << "[GPU] ~ Running total of total size of result array, tid: " << tid << ", " << totalResultsLoop << '\n';
                cout.flush();
            #endif

            double tEndLoop = omp_get_wtime();
            // computeTimeArray[tid] += tEndLoop - tStartLoop;

    	} //END LOOP OVER THE GPU BATCHES

        double computeEndTime = omp_get_wtime();
        computeTime = computeEndTime - computeTimeStart;
        // cout << "[GPU | RESULT] ~ Compute time for the GPU = " << computeEndTime - computeTimeStart << '\n';
        // cout.flush();

        cout << "[BENCH] ~ Compute time for the GPU: " << computeTime << '\n';
        // for(int i = 0; i < GPUSTREAMS; ++i)
        // {
        //     cout << "   [BENCH | Stream " << i << "] ~ Compute time = " << computeTimeArray[i] << ", kernel time = " << kernelTimes[i] << '\n';
        // }

    }
    std::ofstream outfile ("new_mod_neighbor_table.csv",std::ofstream::binary);
    outfile << "pointIdx|originalPointIdx|neighborCnt|neighbors" << endl;
    int neighborCnt = 0;
    for (int i = 0; i < (*DBSIZE); i++) {
        neighborTableLookup tableRecord = neighborTable[originPointIndex[i]];
        neighborCnt = tableRecord.indexmax - tableRecord.indexmin;

        outfile << i << "|" << originPointIndex[i] << "|" << neighborCnt << "|";
        std::vector<unsigned int> neighbors = {};
        for (int j = tableRecord.indexmin; j < tableRecord.indexmax; j++) {
            neighbors.push_back(tableRecord.dataPtr[j]);
        }
        std::sort(neighbors.begin(), neighbors.end());
        for (int j = 0; j < neighborCnt; j++) {
            outfile << neighbors[j];
            if (j == neighborCnt - 1) {
                outfile << endl;
            }
            else {
                outfile << ",";
            }
        }
    }

    unsigned int nbQueryPointTotal = 0;
    for (int i = 0; i < GPUSTREAMS; ++i)
    {
        nbQueryPointTotal += nbQueryPoint[i];
    }

    (*nbQueriesGPU) = nbQueryPointTotal;

    for (int i = 0; i < GPUSTREAMS; ++i)
    {
        printf("[GPU] ~ Kernel execution time on stream %d: %f\n", i, kernelTimes[i]);
    }

    printf("[GPU | RESULT] ~ Query points computed by the GPU: %d (f: %f)\n", nbQueryPointTotal, (nbQueryPointTotal * 1.0) / ((*DBSIZE) * 1.0));
    printf("[GPU | RESULT] ~ Compute time for the GPU: %f\n", computeTime);

    cout << "[GPU] ~ Total result set size on host: " << totalResultsLoop << "\033[00m\n";
    cout.flush();

	*totalNeighbors = totalResultsLoop;

	double tKernelResultsEnd = omp_get_wtime();

    cout << "[GPU] ~ Time to launch kernel and execute everything except freeing memory: " << tKernelResultsEnd - tKernelResultsStart << '\n';
    cout.flush();

	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////





	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
	// if (NUM_TRIALS>1)
	// {

	double tFreeStart = omp_get_wtime();

	for (int i = 0; i < GPUSTREAMS; i++)
    {
		errCode = cudaStreamDestroy(stream[i]);
		if (errCode != cudaSuccess) {
			cout << "[GPU] ~ Error: destroying stream" << errCode << '\n';
            cout.flush();
		}
	}

	delete totalResultSetCnt;
	delete[] cnt;
	delete[] N;

	//free the data on the device

	cudaFree(dev_N);
	cudaFree(dev_cnt);
	// cudaFree(dev_offset);
	// cudaFree(dev_batchNumber);


	//free data related to the individual streams for each batch
	for (int i = 0; i < GPUSTREAMS; i++)
    {
		//free the data on the device
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);

		//free on the host
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);
	}

    // cudaFree(dev_pointIDKey);
    // cudaFree(dev_pointInDistValue);

	//free pinned memory on host
	// cudaFreeHost(pointIDKey); 
	// cudaFreeHost(pointInDistValue);

  	double tFreeEnd = omp_get_wtime();

    cout << "[GPU] ~ Time freeing memory: " << tFreeEnd - tFreeStart << '\n';
    cout.flush();
  	// printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);
  	// }
  	cout << "\n[GPU] ~ ** last error at end of fn batches (could be from freeing memory): " << cudaGetLastError() << "\n\n";
    cout.flush();

} // NDGridIndexGlobal





void warmUpGPU() {
    // initialize all ten integers of a device_vector to 1
  	thrust::device_vector<int> D(10, 1);
  	// set the first seven elements of a vector to 9
  	thrust::fill(D.begin(), D.begin() + 7, 9);
  	// initialize a host_vector with the first five elements of D
  	thrust::host_vector<int> H(D.begin(), D.begin() + 5);
  	// set the elements of H to 0, 1, 2, 3, ...
  	thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D
  	thrust::copy(H.begin(), H.end(), D.begin());
  	// print D
  	for (int i = 0; i < D.size(); i++)
    {
        cout << " D[" << i << "] = " << D[i];
    }

	return;
}





void constructNeighborTableKeyValueWithPtrs(
    int * pointIDKey,
    int * pointInDistValue,
    struct neighborTableLookup * neighborTable,
    int * pointersToNeighbors,
    unsigned int * cnt)
{
    //copy the value data:
  	std::copy(pointInDistValue, pointInDistValue + (*cnt), pointersToNeighbors);

	//Step 1: find all of the unique keys and their positions in the key array
	unsigned int numUniqueKeys = 0;

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key = pointIDKey[0];
	tmp.position = 0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i = 1; i < (*cnt) + 1; i++)
    {
		if (pointIDKey[i - 1] != pointIDKey[i]){
			numUniqueKeys++;
			tmp.key = pointIDKey[i];
			tmp.position = i;
			uniqueKeyData.push_back(tmp);
		}
	}

	//insert into the neighbor table the values based on the positions of
	//the unique keys obtained above.
	for (int i = 0; i < uniqueKeyData.size() - 1; i++)
    {
		int keyElem = uniqueKeyData[i].key;
		neighborTable[keyElem].pointID = keyElem;
		neighborTable[keyElem].indexmin = uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax = uniqueKeyData[i + 1].position;

		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr = pointersToNeighbors;
	}
}

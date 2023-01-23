#include "SortByWorkload.h"
#include "structs.h"
#include "params.h"
#include "kernel.h"
#include "WorkQueue.h"
#include "StaticPartition.h"

#include <iostream>
#include <algorithm>

#include "omp.h"

#include <cuda_runtime.h>
#include <cuda.h>

using std::cout;
using std::endl;

void sortByWorkLoad(
        unsigned int searchMode,
        unsigned int * DBSIZE,
        float * staticPartition,
        uint64_t * totalCandidates,
        struct schedulingCell ** sortedDatabaseTmp,
        DTYPE * epsilon,
        DTYPE ** dev_epsilon,
        DTYPE * database,
        DTYPE ** dev_database,
        struct grid * index,
        struct grid ** dev_index,
        unsigned int * indexLookupArr,
        unsigned int ** dev_indexLookupArr,
        struct gridCellLookup * gridCellLookupArr,
        struct gridCellLookup ** dev_gridCellLookupArr,
        DTYPE * minArr,
        DTYPE ** dev_minArr,
        unsigned int * nCells,
        unsigned int ** dev_nCells,
        unsigned int * nNonEmptyCells,
        unsigned int ** dev_nNonEmptyCells,
        unsigned int ** originPointIndex,
        unsigned int ** dev_originPointIndex)
{

    double tStartSortingCells = omp_get_wtime();

    cudaError_t errCode;

    // struct schedulingCell * sortedDatabaseTmp = new schedulingCell[sizeof(struct schedulingCell) * (*nNonEmptyCells)];
    (*sortedDatabaseTmp) = new schedulingCell[sizeof(struct schedulingCell) * (*nNonEmptyCells)];
    struct schedulingCell * dev_sortedDatabaseTmp;

    cudaEvent_t startKernel, endKernel;
    cudaEventCreate(&startKernel);
    cudaEventCreate(&endKernel);

    errCode = cudaMalloc((void**)&dev_sortedDatabaseTmp, sizeof(struct schedulingCell) * (*nNonEmptyCells));
    if (errCode != cudaSuccess)
    {
        cout << "[SORT] ~ Error: Alloc sortedSet -- error with code " << errCode << '\n';
        cout << "[SORT] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
    }

    // Beginning of the sorting section
    int nbBlock = ((*nNonEmptyCells) / BLOCKSIZE) + 1;
    cout << "[SORT] ~ Using " << nbBlock << " blocks of " << BLOCKSIZE << " threads to sort on the GPU\n";
    cout.flush();

    cudaEventRecord(startKernel);
    sortByWorkLoadGlobal<<<nbBlock, BLOCKSIZE>>>((*dev_database), (*dev_epsilon), (*dev_index),
            (*dev_indexLookupArr), (*dev_gridCellLookupArr), (*dev_minArr), (*dev_nCells),
            (*dev_nNonEmptyCells), dev_sortedDatabaseTmp);
    cudaEventRecord(endKernel);

    errCode = cudaGetLastError();
    cout << "[SORT] ~ SORTING KERNEL LAUNCH RETURN: " << errCode << '\n';
    cout.flush();

    cudaDeviceSynchronize();

    errCode = cudaMemcpy((*sortedDatabaseTmp), dev_sortedDatabaseTmp, sizeof(struct schedulingCell) * (*nNonEmptyCells), cudaMemcpyDeviceToHost);
    if (errCode != cudaSuccess)
    {
        cout << "[SORT] ~ Error: copy sorted cells from the GPU -- error with code " << errCode << '\n';
        cout << "[SORT] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
    }

    cudaEventSynchronize(endKernel);
    float timeKernel = 0;
    cudaEventElapsedTime(&timeKernel, startKernel, endKernel);
    cout << "[SORT] ~ Kernel time to sort by workload: " << timeKernel << '\n';
    cout.flush();

    double tBeginSort = omp_get_wtime();
    std::sort((*sortedDatabaseTmp), (*sortedDatabaseTmp) + (*nNonEmptyCells),
            [](const schedulingCell& a, const schedulingCell& b){ return a.nbPoints > b.nbPoints; });
    double tEndSort = omp_get_wtime();
    printf("[SORT] ~ Time to call std::sort = %f\n", tEndSort - tBeginSort);

    unsigned int maxNeighbor = (*sortedDatabaseTmp)[0].nbPoints;
    unsigned int minNeighbor = (*sortedDatabaseTmp)[(*nNonEmptyCells) - 1].nbPoints;
    // cout << "max = " << maxNeighbor << '\n';
    // cout << "min = " << minNeighbor << '\n';
    uint64_t accNeighbor = 0;

    (*originPointIndex) = new unsigned int [(*DBSIZE)];

    uint64_t partitionGPU = 0;
    // Find the amount of candidate points to attribute to the GPU (and therefore to the CPU)
    if (SM_HYBRID_STATIC == searchMode)
    {
        #if !STATIC_SPLIT_QUERIES
            for (int i = 0; i < (*nNonEmptyCells); ++i)
            {
                int cellId = (*sortedDatabaseTmp)[i].cellId;
                int nbNeighbor = index[cellId].indexmax - index[cellId].indexmin + 1;
                partitionGPU += (nbNeighbor * (*sortedDatabaseTmp)[i].nbPoints);
            }

            if(-1.0 == (*staticPartition))
            {
                // If the user didn't specify a static partitioning value at runtime, we use our model
                // Put the model to estimate the partition here, as explained in main.cu
                double gpuTimeModel = getGPUTimeCandidates((*DBSIZE), (*epsilon), partitionGPU);
                double cpuTimeModel = getCPUTimeCandidates((*DBSIZE), (*epsilon), partitionGPU);
                fprintf(stdout, "[MODEL] ~ Times before un-logging: GPU = %f, CPU = %f\n", gpuTimeModel, cpuTimeModel);
                // gpuTimeModel and cpuTimeModel are on a log10 scale, so un-log them
                gpuTimeModel = exp(gpuTimeModel);
                cpuTimeModel = exp(cpuTimeModel);

                uint64_t gpu_cps = partitionGPU / gpuTimeModel;
                uint64_t cpu_cps = partitionGPU / cpuTimeModel;
                uint64_t upper_cps = gpu_cps + cpu_cps;

                double theoreticalTime = (partitionGPU * 1.0) / (upper_cps * 1.0);

                (*staticPartition) = (gpu_cps * 1.0) / (upper_cps * 1.0);

                fprintf(stdout, "[MODEL | RESULT] ~ GPU time: %f, CPU time: %f, theoretical time: %f\n", gpuTimeModel, cpuTimeModel, theoreticalTime);
                fprintf(stdout, "[MODEL | RESULT] ~ GPU candidates/s: %lu, CPU candidates/s: %lu, upper candidates/s: %lu\n", gpu_cps, cpu_cps, upper_cps);
                fprintf(stdout, "[MODEL | RESULT] ~ Modeled GPU partition: %f, CPU partition: %f\n", (*staticPartition), 1 - (*staticPartition));
            } else {
                // Nothing particular to do, we already know the static partitioning
            }
        #endif
    }
    partitionGPU *= (*staticPartition);
    bool partitioned = false;
    int partitionPoint = 0;

    int prec = 0;
    uint64_t runningPartition = 0;
    for (int i = 0; i < (*nNonEmptyCells); ++i)
    {
        int cellId = (*sortedDatabaseTmp)[i].cellId;
        int nbNeighbor = index[cellId].indexmax - index[cellId].indexmin + 1;

        accNeighbor += ((uint64_t)nbNeighbor * (uint64_t)(*sortedDatabaseTmp)[i].nbPoints);

        for (int j = 0; j < nbNeighbor; ++j)
        {
            int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
            (*originPointIndex)[prec + j] = tmpId;
            runningPartition += (*sortedDatabaseTmp)[i].nbPoints;

            // Find the query point that statically partition the candidate points between the CPU and the GPU
            if (SM_HYBRID_STATIC == searchMode)
            {
                #if !STATIC_SPLIT_QUERIES
                    if (!partitioned && partitionGPU <= runningPartition)
                    {
                        partitionPoint = prec + j;
                        partitioned = true;
                    }
                #endif
            }
        }
        prec += nbNeighbor;
    }

    // Set the query point that statically partition the candidate points between the CPU and the GPU
    if (SM_HYBRID_STATIC == searchMode)
    {
        #if !STATIC_SPLIT_QUERIES
            cout << "[SORT | DEBUG] ~ Setting the number of query points for the GPU: " << partitionPoint << '\n';
            setStaticQueryPoint(partitionPoint);
        #endif
    }

    // Setting some stuff for the CPU so it can begin immediately
    // and allocate buffers to store temp results
    // However, this value is way overestimated as it is set to the max,
    // while the CPU computes from the end of the queue.
    setMaxNeighbors(maxNeighbor);
    setWorkQueueReady();

    // errCode = cudaMalloc((void**)dev_originPointIndex, ((*DBSIZE) - nbQueriesPreComputed) * sizeof(unsigned int));
    errCode = cudaMalloc((void**)dev_originPointIndex, (*DBSIZE) * sizeof(unsigned int));
    if (errCode != cudaSuccess)
    {
        cout << "[SORT] ~ Error: Alloc point index -- error with code " << errCode << '\n';
        cout << "[SORT] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
    }

    // errCode = cudaMemcpy( (*dev_originPointIndex), (*originPointIndex), ((*DBSIZE) - nbQueriesPreComputed) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    errCode = cudaMemcpy( (*dev_originPointIndex), (*originPointIndex), (*DBSIZE) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (errCode != cudaSuccess)
    {
        cout << "[SORT] ~ Error: point index copy -- error with code " << errCode << '\n';
        cout << "[SORT] ~   Details: " << cudaGetErrorString(errCode) << '\n';
        cout.flush();
    }

    (*totalCandidates) = accNeighbor;

    unsigned int decileMark = (*nNonEmptyCells) / 10;
    cout << "[SORT | RESULT] ~ Total number of candidate points to refine: " << accNeighbor << '\n';
    cout << "[SORT | RESULT] ~ Number of candidates: min = " << minNeighbor << ", median = " << (*sortedDatabaseTmp)[(*nNonEmptyCells) / 2].nbPoints << ", max = " << maxNeighbor << ", avg = " << accNeighbor / (*DBSIZE) << '\n';
    cout << "[SORT] ~ Deciles number of candidates: \n";
    for (int i = 1; i < 10; ++i)
    {
        cout << "   [SORT] ~ " << i * 10 << "% = " << (*sortedDatabaseTmp)[decileMark * i].nbPoints << '\n';
    }
    cout.flush();

    if (SM_HYBRID_STATIC == searchMode)
    {
        #if !STATIC_SPLIT_QUERIES
            printf("[SORT | RESULT] ~ %u query points assigned to the GPU, with %llu candidates to refines\n", partitionPoint, runningPartition);
            printf("[SORT | RESULT] ~ %u query points assigned to the CPU, with %llu candidates to refines\n", (*DBSIZE) - partitionPoint, accNeighbor - runningPartition);
        #endif
    }

    cudaFree(dev_sortedDatabaseTmp);

    // delete[] sortedDatabaseTmp;
    // delete[] nbNeighborPoints;

    double tEndSortingCells = omp_get_wtime();

    cout << "[SORT] ~ Time to sort the cells by workload and copy to the GPU: " << tEndSortingCells - tStartSortingCells << '\n';
    cout.flush();

}

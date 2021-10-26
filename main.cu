#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <set>
#include <algorithm>

#include "omp.h"

#include "main.h"
#include "params.h"
#include "structs.h"
#include "SortByWorkload.h"
#include "WorkQueue.h"
#include "GPU.h"
#include "StaticPartition.h"

#include "Point.hpp"
#include "Util.hpp"

// #include <boost/sort/sort.hpp>
#include <boost/sort/sort.hpp>
// #include <parallel/algorithm>

using std::cout;
using std::endl;


bool egoSortFunction(Point const& p1, Point const& p2)
{
    for (int i = 0; i < GPUNUMDIM; i++)
	{
		if ( (int) (p1.x[i] / Util::eps) < (int)(p2.x[i] / Util::eps) ) return true;
		if ( (int) (p1.x[i] / Util::eps) > (int)(p2.x[i] / Util::eps) ) return false;
	}

	return false;
}


int main(int argc, char * argv[])
{
    double tStartStart = omp_get_wtime();
    printf("\n\n========== Hybrid-EGO ==========\n\n\n");

    if(NB_ARGS_MAX != argc && (NB_ARGS_MAX - 1) != argc)
    {
        fprintf(stderr, "[MAIN] ~ Expected %d or %d args, found %d\n", NB_ARGS_MAX, NB_ARGS_MAX - 1, argc);
        fprintf(stderr, "[MAIN] ~ Args: filename epsilon dim searchmode [CPU-GPU partitioning]\n");
        return 1;
    }

    char filename[256];
    strcpy(filename, argv[FILENAME_ARG]);
    DTYPE epsilon = atof(argv[EPSILON_ARG]);
    int dim = atoi(argv[DIM_ARG]);
    int searchMode = atoi(argv[SEARCHMODE_ARG]);
    float staticPartition = -1.0; // Set it as a default value for later

    if(GPUNUMDIM != dim)
    {
        fprintf(stderr, "[MAIN] ~ Error: dim should be equals to GPUNUMDIM (see params.h)\n");
        return 1;
    }

    if(epsilon <= 0.0 || 1.0 < epsilon)
    {
        fprintf(stderr, "[MAIN] ~ Error: epsilon should be between ]0.0, 1.0]\n");
        return 1;
    }

    fprintf(stdout, "[MAIN] ~ Dataset: %s\n", filename);
    fprintf(stdout, "[MAIN] ~ Epsilon: %f\n", epsilon);
    fprintf(stdout, "[MAIN] ~ Dimensionality: %d\n", dim);
    fprintf(stdout, "[MAIN] ~ Search mode: %d\n", searchMode);
    // if (SM_HYBRID_STATIC == searchMode)
    // {
    //     fprintf(stdout, "[MAIN] ~ GPU part: %f, CPU part: %f\n", staticPartition, 1 - staticPartition);
    // }

    Util::eps = epsilon;
    Util::eps2 = epsilon * epsilon;

    std::vector< std::vector<DTYPE> > NDdataPoints;
    double tBeginReadDataset = omp_get_wtime();
    // DTYPE * database;
    // unsigned int nbPoints = 0;
    importNDDataset(&NDdataPoints, filename);
    // importNDDatasetBinary(&NDdataPoints, filename);
    // importNDDatasetBinary(&database, filename, &nbPoints);
    double tEndReadDataset = omp_get_wtime();
    fprintf(stdout, "[MAIN] ~ Time to read the dataset: %f\n", tEndReadDataset - tBeginReadDataset);

    unsigned int DBSIZE = NDdataPoints.size();
    // unsigned int DBSIZE = nbPoints;
    setQueueIndexCPU(DBSIZE);

    // sortInNDBins(&NDdataPoints);

    Point * A;
    Point * B;
    if(SM_GPU != searchMode)
    {
        fprintf(stdout, "[MAIN] ~ Converting the dataset for Super-EGO\n");
        A = new Point[DBSIZE + 1];
        for(int i = 0; i < DBSIZE; ++i)
        {
            pPoint p = & A[i];
            p->id = i;
            for(int j = 0; j < GPUNUMDIM; ++j)
            {
                p->x[j] = NDdataPoints[i][j];
                // p->x[j] = database[i * GPUNUMDIM + j];
            }
        }
        B = A;
    }

    DTYPE * database = new DTYPE [DBSIZE * GPUNUMDIM];
    for(int i = 0; i < DBSIZE; ++i)
    {
        for(int j = 0; j < GPUNUMDIM; ++j)
        {
            database[i * GPUNUMDIM + j] = NDdataPoints[i][j];
        }
        // std::copy(NDdataPoints[i].begin(), NDdataPoints[i].end(), database + i * GPUNUMDIM);
    }

    // Static partitioning between CPU and GPU components
    if (SM_HYBRID_STATIC == searchMode)
    {
        if (NB_ARGS_MAX == argc)
        {
            // Read the static partition from command line
            staticPartition = atof(argv[STATIC_PART_ARG]);
            if (staticPartition <= 0.0 || 1.0 <= staticPartition)
            {
                fprintf(stderr, "[MAIN] ~ Error: the partitioning should be between ]0.0, 1.0[");
                return 1;
            }
            fprintf(stdout, "[MODEL | RESULT] ~ Statically partitioning the work, using the partitioning value %f\n", staticPartition);
        } else {
            // The static partition was not given, so estimate it following our model and other parameters
            #if STATIC_SPLIT_QUERIES
                double gpuTimeModel = getGPUTimeQueries(DBSIZE, epsilon);
                double cpuTimeModel = getCPUTimeQueries(DBSIZE, epsilon);
                fprintf(stdout, "[MODEL] ~ Times before un-logging: GPU = %f, CPU = %f\n", gpuTimeModel, cpuTimeModel);
                // gpuTimeModel and cpuTimeModel are on a log10 scale, so un-log them
                gpuTimeModel = exp(gpuTimeModel);
                cpuTimeModel = exp(cpuTimeModel);

                int gpu_qps = DBSIZE / gpuTimeModel;
                int cpu_qps = DBSIZE / cpuTimeModel;
                int upper_qps = gpu_qps + cpu_qps;

                double theoreticalTime = (DBSIZE * 1.0) / (upper_qps * 1.0);

                staticPartition = (gpu_qps * 1.0) / (upper_qps * 1.0);

                fprintf(stdout, "[MODEL | RESULT] ~ GPU time: %f, CPU time: %f, theoretical time: %f\n", gpuTimeModel, cpuTimeModel, theoreticalTime);
                fprintf(stdout, "[MODEL | RESULT] ~ GPU queries/s: %d, CPU queries/s: %d, upper queries/s: %d\n", gpu_qps, cpu_qps, upper_qps);
                fprintf(stdout, "[MODEL | RESULT] ~ Modeled GPU partition: %f, CPU partition: %f\n", staticPartition, 1 - staticPartition);
            #endif

            ////////////////////////////////////////////////////////////////////////////////////////
            //
            // The model to partition based on the candidates is inside the SortByWL function,
            // as SortByWL needs to know the partition to determine which query point partitions the work,
            // and the model needs SortByWL to know how many candidates there are.
            //
            ////////////////////////////////////////////////////////////////////////////////////////

            // double gpuTimeModel = getGPUTimeCandidates(DBSIZE, epsilon, totalCandidates);
            // double cpuTimeModel = getCPUTimeCandidates(DBSIZE, epsilon, totalCandidates);
            //
            // uint64_t gpu_cps = totalCandidates / gpuTimeModel;
            // uint64_t cpu_cps = totalCandidates / cpuTimeModel;
            // uint64_t upper_cps = gpu_cps + cpu_cps;
            //
            // staticPartition = gpu_cps / upper_cps;
            //
            // fprintf(stdout, "[MODEL] ~ GPU time: %f, CPU time: %f\n", gpuTimeModel, cpuTimeModel);
            // fprintf(stdout, "[MODEL] ~ GPU queries/s: %lu, CPU queries/s: %lu, upper queries/s: %lu\n", gpu_cps, cpu_cps, upper_cps);
            // fprintf(stdout, "[MODEL] ~ Modeled GPU partition: %f, CPU partition: %f\n", staticPartition, 1 - staticPartition);
            // #endif
        }
    } else { // Dynamic partitioning or CPU/GPU alone
        if (SM_GPU == searchMode)
        {
            // The GPU is alone so it takes all the work
            staticPartition = 1.0;
        } else {
            // The CPU is alone so it takes all the work, or it's the regular (dynamic) hybrid
            // so the staticPartition value does not matter
            staticPartition = 0.0;
        }
    }

    DTYPE * minArr = new DTYPE [NUMINDEXEDDIM];
    DTYPE * maxArr = new DTYPE [NUMINDEXEDDIM];
    unsigned int * nCells = new unsigned int [NUMINDEXEDDIM];
    uint64_t totalCells = 0;
    unsigned int nNonEmptyCells = 0;

    generateNDGridDimensions(&NDdataPoints, epsilon, minArr, maxArr, nCells, &totalCells);
    fprintf(stdout, "[MAIN] ~ Total cells (including empty): %lu\n", totalCells);

    struct grid * index;
    struct gridCellLookup * gridCellLookupArr;
    unsigned int * indexLookupArr = new unsigned int[NDdataPoints.size()];

    DTYPE * dev_epsilon;
    DTYPE * dev_database;
    struct grid * dev_index;
    unsigned int * dev_indexLookupArr;
    struct gridCellLookup * dev_gridCellLookupArr;
    DTYPE * dev_minArr;
    unsigned int * dev_nCells;
    unsigned int * dev_nNonEmptyCells;

    gridIndexingGPU(&DBSIZE, totalCells, database, &dev_database, &epsilon, &dev_epsilon, minArr, &dev_minArr, &index, &dev_index,
            indexLookupArr, &dev_indexLookupArr, &gridCellLookupArr, &dev_gridCellLookupArr, &nNonEmptyCells, &dev_nNonEmptyCells,
            nCells, &dev_nCells);

    //Neighbortable storage -- the result
    neighborTableLookup * neighborTable = new neighborTableLookup[NDdataPoints.size()];
    std::vector<struct neighborDataPtrs> pointersToNeighbors;

    unsigned int * originPointIndex;
    unsigned int * dev_originPointIndex;

    uint64_t totalNeighbors = 0;
    uint64_t totalNeighborsCPU = 0;
    uint64_t totalCandidates = 0;

    struct schedulingCell * sortedDatabaseTmp;

    double sortTime, gpuTime, egoTime, egoReorder, egoSort;

    double tStartSort = omp_get_wtime();
    #if SORT_BY_WORKLOAD
        sortByWorkLoad(searchMode, &DBSIZE, &staticPartition, &totalCandidates, &sortedDatabaseTmp, &epsilon, &dev_epsilon,
                database, &dev_database, index, &dev_index, indexLookupArr, &dev_indexLookupArr,
                gridCellLookupArr, &dev_gridCellLookupArr, minArr, &dev_minArr, nCells, &dev_nCells,
                &nNonEmptyCells, &dev_nNonEmptyCells, &originPointIndex, &dev_originPointIndex);
    #endif
    double tEndSort = omp_get_wtime();
    sortTime = tEndSort - tStartSort;

    fprintf(stdout, "\n\n[MAIN] ~ Time to do everything before computing: %f\n\n\n", tEndSort - tStartStart);

    unsigned int nbQueriesGPU = 0;

    omp_set_nested(1);
	  omp_set_dynamic(0);

    double tStart = omp_get_wtime();
    double tEndGPU, tEndEgo;
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();

        if(0 == tid) // GPU part
        {
            if(searchMode != SM_CPU)
            {
                double tBeginGPU = omp_get_wtime();
                #if SORT_BY_WORKLOAD
                    distanceTableNDGridBatches(searchMode, staticPartition, &DBSIZE, &epsilon, dev_epsilon, database, dev_database,
                            index, dev_index, indexLookupArr, dev_indexLookupArr, gridCellLookupArr, dev_gridCellLookupArr,
                            minArr, dev_minArr, nCells, dev_nCells, &nNonEmptyCells, dev_nNonEmptyCells,
                            originPointIndex, dev_originPointIndex, neighborTable, &pointersToNeighbors, &totalNeighbors, &nbQueriesGPU);
                #else
                    distanceTableNDGridBatches(searchMode, staticPartition, &DBSIZE, &epsilon, dev_epsilon, database, dev_database,
                            index, dev_index, indexLookupArr, dev_indexLookupArr, gridCellLookupArr, dev_gridCellLookupArr,
                            minArr, dev_minArr, nCells, dev_nCells, &nNonEmptyCells, dev_nNonEmptyCells,
                            nullptr, nullptr, neighborTable, &pointersToNeighbors, &totalNeighbors, &nbQueriesGPU);
                #endif
                tEndGPU = omp_get_wtime();
                gpuTime = tEndGPU - tBeginGPU;
            }
        }
        else // Super-EGO part
        {
            if(searchMode != SM_GPU)
            {
                if(searchMode == SM_CPU)
                {
                    setQueueIndex(0);
                }

                unsigned int A_sz = DBSIZE;
                unsigned int B_sz = DBSIZE;

                fprintf(stdout, "[EGO] ~ Reordering the dimensions\n");
                double tStartReorder = omp_get_wtime();
                Util::reorderDim(A, A_sz, B, B_sz);
                double tEndReorder = omp_get_wtime();
                egoReorder = tEndReorder - tStartReorder;
                fprintf(stdout, "[EGO] ~ Done reordering in %f\n", egoReorder);

                fprintf(stdout, "[EGO] ~ EGO-sorting of A\n");
                double tStartEGOSort = omp_get_wtime();
                // std::stable_sort(A, A + A_sz, egoSortFunction);
                boost::sort::sample_sort(A, A + A_sz, egoSortFunction, CPU_THREADS);
                double tEndEGOSort = omp_get_wtime();
                egoSort = tEndEGOSort - tStartEGOSort;
                fprintf(stdout, "[EGO] ~ Done EGO-sorting in %f\n", egoSort);

                fprintf(stdout, "[EGO] ~ Creating the mapping\n");
                double tStartMapping = omp_get_wtime();
                unsigned int * egoMapping = new unsigned int[DBSIZE];
                for(int i = 0; i < DBSIZE; ++i)
                {
                    pPoint p = &A[i];
                    egoMapping[p->id] = i;
                }
                double tEndMapping = omp_get_wtime();
                fprintf(stdout, "[EGO] ~ Done creating the mapping in %f\n", tEndMapping - tStartMapping);

                double tBeginEgo = omp_get_wtime();

                uint64_t * nbCandidatesArray = new uint64_t[DBSIZE];
                for (int i = 0; i < DBSIZE; ++i)
                {
                    nbCandidatesArray[i] = 0;
                }

                fprintf(stdout, "[EGO] ~ Beginning the computation\n");
                #if SORT_BY_WORKLOAD
                    totalNeighborsCPU = Util::multiThreadJoinWorkQueue(searchMode, A, A_sz, B, B_sz, egoMapping, originPointIndex, neighborTable, nbCandidatesArray);
                #else
                    totalNeighborsCPU = Util::multiThreadJoinWorkQueue(searchMode, A, A_sz, B, B_sz, egoMapping, nullptr, neighborTable, nbCandidatesArray);
                #endif
                fprintf(stdout, "[EGO] ~ Done with the computation\n");

                tEndEgo = omp_get_wtime();
                egoTime = tEndEgo - tBeginEgo;

                #if COMPARE_CANDIDATES
                    for (int i = 0; i < nNonEmptyCells; ++i)
                    {
                        int cellId = sortedDatabaseTmp[i].cellId;
                        int nbPoints = index[cellId].indexmax - index[cellId].indexmin + 1;
                        uint64_t nbCandidates = sortedDatabaseTmp[i].nbPoints;

                        for (int j = 0; j < nbPoints; ++j)
                        {
                            int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
                            int tmpIdEgo = egoMapping[tmpId];
                            pPoint tmpPoint = &A[tmpIdEgo];
                            printf("nb_candidates, %d, %lu, %lu\n", tmpId, nbCandidates, nbCandidatesArray[tmpPoint->id]);
                        }
                    }
                #endif
                delete[] nbCandidatesArray;
                delete[] egoMapping;
            } // searchMode
        } // Super-EGO
        #pragma omp barrier
    } // parallel section
    double tEnd = omp_get_wtime();
    double computeTime = tEnd - tStart;

    displayIndexes();

    fprintf(stdout, "[RESULT] ~ Total result set size: %lu\n", totalNeighbors + totalNeighborsCPU);
    fprintf(stdout, "   [RESULT] ~ Total result set size on the GPU: %lu\n", totalNeighbors);
    fprintf(stdout, "   [RESULT] ~ Total result set size on the CPU: %lu\n", totalNeighborsCPU);

    #if COUNT_CANDIDATES_GPU
        if (searchMode == SM_HYBRID || searchMode == SM_HYBRID_STATIC)
        {
            uint64_t nbCandidatesGPU = 0;
            unsigned int nbQueriesTmp = 0;
            for (int i = 0; i < nNonEmptyCells; ++i)
            {
                int cellId = sortedDatabaseTmp[i].cellId;
                int nbPoints = index[cellId].indexmax - index[cellId].indexmin + 1;
                if (nbQueriesGPU < nbQueriesTmp + nbPoints)
                {
                    nbCandidatesGPU += (nbQueriesGPU - nbQueriesTmp) * sortedDatabaseTmp[i].nbPoints;
                    break;
                } else {
                    nbCandidatesGPU += (nbPoints * sortedDatabaseTmp[i].nbPoints);
                    nbQueriesTmp += nbPoints;
                }
            }
            fprintf(stdout, "   [RESULT] ~ Total number of candidate points refined by the GPU: %lu (f: %f)\n", nbCandidatesGPU, (nbCandidatesGPU * 1.0) / (totalCandidates * 1.0));
            fprintf(stdout, "   [RESULT] ~ Total number of candidate points refined by the CPU (if using the grid): %lu\n", totalCandidates - nbCandidatesGPU);
        }
    #endif

    fprintf(stdout, "[RESULT] ~ Total execution time: %f\n", computeTime + sortTime);
    fprintf(stdout, "   [RESULT] ~ Total execution time to SortByWL: %f\n", sortTime);
    fprintf(stdout, "   [RESULT] ~ Total execution time for the GPU: %f\n", gpuTime);
    fprintf(stdout, "   [RESULT] ~ Total execution time for the CPU: %f (reorder: %f, sort: %f, total = %f)\n", egoTime, egoReorder, egoSort, egoTime + egoReorder + egoSort);

    if(tEndGPU < tEndEgo)
    {
        double timeDiff = tEndEgo - tEndGPU;
        fprintf(stdout, "[RESULT] ~ The GPU ended before the CPU, with a difference of: %f (ratio: %f)\n", timeDiff, timeDiff / (computeTime + sortTime));
    }else{
        double timeDiff = tEndGPU - tEndEgo;
        fprintf(stdout, "[RESULT] ~ The CPU ended before the GPU, with a difference of: %f (ratio: %f)\n", timeDiff, timeDiff / (computeTime + sortTime));
    }

    // printNeighborTable(neighborTable, 0, 20);

    NDdataPoints.clear();
    NDdataPoints.shrink_to_fit();
    pointersToNeighbors.clear();
    pointersToNeighbors.shrink_to_fit();

    delete[] minArr;
    delete[] maxArr;
    delete[] nCells;
    delete[] index;
    delete[] sortedDatabaseTmp;
    delete[] gridCellLookupArr;
    delete[] indexLookupArr;
    delete[] neighborTable;
    delete[] database;

    if(SM_GPU != searchMode)
    {
        delete[] A;
    }

    cudaFree(dev_epsilon);
    cudaFree(dev_database);
    cudaFree(dev_index);
    cudaFree(dev_indexLookupArr);
    cudaFree(dev_gridCellLookupArr);
    cudaFree(dev_minArr);
    cudaFree(dev_nCells);
    cudaFree(dev_nNonEmptyCells);

    #if SORT_BY_WORKLOAD
        delete[] originPointIndex;
        cudaFree(dev_originPointIndex);
    #endif

    return 0;
}


void generateNDGridDimensions(
        std::vector< std::vector <DTYPE> > * NDdataPoints,
        DTYPE epsilon,
        DTYPE* minArr,
        DTYPE* maxArr,
        unsigned int * nCells,
        uint64_t * totalCells)
{

    printf("\n\n*****************************  Generating grid dimensions  *****************************\n");

    printf("\nNumber of dimensions data: %d, Number of dimensions indexed: %d", GPUNUMDIM, NUMINDEXEDDIM);

    //make the min/max values for each grid dimension the first data element
    for(int j = 0; j < NUMINDEXEDDIM; j++)
    {
        minArr[j] = (*NDdataPoints)[0][j];
        maxArr[j] = (*NDdataPoints)[0][j];
    }



    for(int i = 1; i < NDdataPoints->size(); ++i)
    {
        for(int j = 0; j < NUMINDEXEDDIM; j++)
        {
            if ((*NDdataPoints)[i][j] < minArr[j])
            {
                minArr[j] = (*NDdataPoints)[i][j];
            }
            if((*NDdataPoints)[i][j] > maxArr[j])
            {
                maxArr[j] = (*NDdataPoints)[i][j];
            }
        }
    }

    printf("\n");
    for(int j = 0; j < NUMINDEXEDDIM; j++)
    {
        printf("Data Dim: %d, min/max: %f, %f\n", j, minArr[j], maxArr[j]);
    }

    //add buffer around each dim so no weirdness later with putting data into cells
    for(int j = 0; j < NUMINDEXEDDIM; j++)
    {
        minArr[j] -= epsilon;
        maxArr[j] += epsilon;
    }

    for(int j = 0; j < NUMINDEXEDDIM; j++)
    {
        printf("Appended by epsilon Dim: %d, min/max: %f, %f\n", j, minArr[j], maxArr[j]);
    }

    //calculate the number of cells:
    for(int j = 0; j < NUMINDEXEDDIM; j++)
    {
        nCells[j] = ceil((maxArr[j] - minArr[j]) / epsilon);
        printf("Number of cells dim: %d: %d\n", j, nCells[j]);
    }

    //calc total cells: num cells in each dim multiplied
    uint64_t tmpTotalCells = nCells[0];
    for(int j = 1; j < NUMINDEXEDDIM; j++)
    {
        tmpTotalCells *= nCells[j];
    }

    *totalCells = tmpTotalCells;

}


void printNeighborTable(
    struct neighborTableLookup * neighborTable,
    unsigned int begin,
    unsigned int end)
{
    printf("\n");
    for (int i = begin; i < end; ++i)
    {
	 	// sort to compare against CPU implementation
	 	std::sort(neighborTable[i].dataPtr + neighborTable[i].indexmin, neighborTable[i].dataPtr + neighborTable[i].indexmax + 1);
	 	printf("point id: %d, neighbors: %d\n", i, neighborTable[i].indexmax - neighborTable[i].indexmin);
	 	for (int j = neighborTable[i].indexmin; j < neighborTable[i].indexmax - 1; j++)
    {
	 		  printf("%d, ", neighborTable[i].dataPtr[j]);
	 	}
        printf("%d\n", neighborTable[i].dataPtr[ neighborTable[i].indexmax - 1 ]);
    }
}


struct cmpStruct
{
    cmpStruct(std::vector< std::vector<DTYPE> > points) {this->points = points;}
    bool operator() (int a, int b)
    {
        return points[a][0] < points[b][0];
    }

    std::vector< std::vector<DTYPE> > points;
};


void populateNDGridIndexAndLookupArray(
        std::vector<std::vector <DTYPE> > * NDdataPoints,
        DTYPE epsilon,
        struct gridCellLookup ** gridCellLookupArr,
        struct grid ** index,
        unsigned int * indexLookupArr,
        DTYPE* minArr,
        unsigned int * nCells,
        uint64_t totalCells,
        unsigned int * nNonEmptyCells,
        unsigned int ** gridCellNDMask,
        unsigned int * gridCellNDMaskOffsets,
        unsigned int * nNDMaskElems)
{

    /////////////////////////////////
    // Populate grid lookup array
    // and corresponding indicies in the lookup array
    /////////////////////////////////
    printf("\n\n*****************************  Populating Grid Index and lookup array  *****************************\n");
    // printf("\nSize of dataset: %lu", NDdataPoints->size());


    ///////////////////////////////
    // First, we need to figure out how many non-empty cells there will be
    // For memory allocation
    // Need to do a scan of the dataset and calculate this
    // Also need to keep track of the list of uniquie linear grid cell IDs for inserting into the grid
    ///////////////////////////////
    std::set<uint64_t> uniqueGridCellLinearIds;
    std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

    for (int i = 0; i < NDdataPoints->size(); ++i)
    {
        unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
        for (int j = 0; j < NUMINDEXEDDIM; j++)
        {
            tmpNDCellIdx[j] = (((*NDdataPoints)[i][j] - minArr[j]) / epsilon);
        }
        uint64_t linearID = getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
        uniqueGridCellLinearIds.insert(linearID);

    }

    // printf("uniqueGridCellLinearIds: %d",uniqueGridCellLinearIds.size());

    //copy the set to the vector (sets can't do binary searches -- no random access)
    std::copy(uniqueGridCellLinearIds.begin(), uniqueGridCellLinearIds.end(), std::back_inserter(uniqueGridCellLinearIdsVect));


    ///////////////////////////////////////////////


    std::vector<uint64_t> * gridElemIDs;
    gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIds.size()];

    //Create ND array mask:
    //This mask determines which cells in each dimension has points in them.
    std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];

    std::vector<uint64_t>::iterator lower;


    for (int i = 0; i < NDdataPoints->size(); i++)
    {
        unsigned int tmpNDCellID[NUMINDEXEDDIM];
        for (int j = 0; j < NUMINDEXEDDIM; j++)
        {
            tmpNDCellID[j] = (((*NDdataPoints)[i][j] - minArr[j]) / epsilon);

            //add value to the ND array mask
            NDArrMask[j].insert(tmpNDCellID[j]);
        }

        //get the linear id of the cell
        uint64_t linearID = getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
        //printf("\nlinear id: %d",linearID);
        //if (linearID > totalCells){

        //	printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
        //}

        //find the index in gridElemIds that corresponds to this grid cell linear id

        lower = std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(), linearID);
        uint64_t gridIdx = lower - uniqueGridCellLinearIdsVect.begin();
        gridElemIDs[gridIdx].push_back(i);
    }




    ///////////////////////////////
    //Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
    ///////////////////////////////

    struct grid * tmpIndex = new grid[uniqueGridCellLinearIdsVect.size()];

    int cnt = 0;



    //populate temp index and lookup array
    for (int i = 0; i < uniqueGridCellLinearIdsVect.size(); i++)
    {
        tmpIndex[i].indexmin = cnt;
        for (int j = 0; j < gridElemIDs[i].size(); j++)
        {
            if ((NDdataPoints->size() - 1) < j)
            {
                printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
                return;
            }
            indexLookupArr[cnt] = gridElemIDs[i][j];
            cnt++;
        }
        tmpIndex[i].indexmax = cnt - 1;
    }

    // printf("\nExiting grid populate method early!");
    // return;

    printf("\nFull cells: %d (%f, fraction full)", (unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size() * 1.0 / double(totalCells));
    printf("\nEmpty cells: %ld (%f, fraction empty)", totalCells - (unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells - uniqueGridCellLinearIdsVect.size() * 1.0) / double(totalCells));

    *nNonEmptyCells = uniqueGridCellLinearIdsVect.size();

    printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid) * (totalCells) / (1024.0 * 1024.0 * 1024.0));
    printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid) * (uniqueGridCellLinearIdsVect.size() * 1.0) / (1024.0 * 1024.0 * 1024.0));


    /////////////////////////////////////////
    //copy the tmp index into the actual index that only has the non-empty cells
    //allocate memory for the index that will be sent to the GPU
    *index = new grid[uniqueGridCellLinearIdsVect.size()];
    *gridCellLookupArr = new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

    cmpStruct theStruct(*NDdataPoints);

    for (int i = 0; i < uniqueGridCellLinearIdsVect.size(); i++)
    {
        (*index)[i].indexmin = tmpIndex[i].indexmin;
        (*index)[i].indexmax = tmpIndex[i].indexmax;
        (*gridCellLookupArr)[i].idx = i;
        (*gridCellLookupArr)[i].gridLinearID = uniqueGridCellLinearIdsVect[i];
    }

    printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu", uniqueGridCellLinearIdsVect.size());

    //copy NDArrMask from set to an array

    //find the total size and allocate the array
    unsigned int cntNDOffsets = 0;
    unsigned int cntNonEmptyNDMask = 0;
    for (int i = 0; i < NUMINDEXEDDIM; i++)
    {
        cntNonEmptyNDMask += NDArrMask[i].size();
    }
    *gridCellNDMask = new unsigned int[cntNonEmptyNDMask];

    *nNDMaskElems = cntNonEmptyNDMask;


    //copy the offsets to the array
    for (int i = 0; i < NUMINDEXEDDIM; i++)
    {
        //Min
        gridCellNDMaskOffsets[(i * 2)] = cntNDOffsets;
        for (std::set<unsigned int>::iterator it = NDArrMask[i].begin(); it != NDArrMask[i].end(); ++it)
        {
            (*gridCellNDMask)[cntNDOffsets] = *it;
            cntNDOffsets++;
        }
        //max
        gridCellNDMaskOffsets[(i * 2) + 1]  =cntNDOffsets - 1;
    }

    delete [] tmpIndex;
}


uint64_t getLinearID_nDimensions(
        unsigned int * indexes,
        unsigned int * dimLen,
        unsigned int nDimensions)
{
    uint64_t index = 0;
    uint64_t multiplier = 1;
    for (int i = 0; i < nDimensions; i++)
    {
        index += (uint64_t)indexes[i] * multiplier;
        multiplier *= dimLen[i];
    }

    return index;
}

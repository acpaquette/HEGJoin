#ifndef GPU_H
#define GPU_H

#include "structs.h"
#include "params.h"

#include <vector>

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
    unsigned int ** dev_nCells);

void distanceTableNDGridBatches(
        int searchMode,
        unsigned int * DBSIZE,
        DTYPE * epsilon,
        DTYPE * dev_epsilon,
        DTYPE * database,
        DTYPE * dev_database,
        struct grid * index,
        struct grid * dev_index,
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
        unsigned int * originPointIndex,
        unsigned int * dev_originPointIndex,
        struct neighborTableLookup * neighborTable,
        std::vector<struct neighborDataPtrs> * pointersToNeighbors,
        uint64_t * totalNeighbors);

unsigned long long callGPUBatchEst(
        unsigned int * DBSIZE,
        unsigned int sampleBegin,
        unsigned int sampleEnd,
        DTYPE * dev_database,
        DTYPE * dev_sortedDatabase,
        unsigned int * dev_originPointIndex,
        DTYPE * dev_epsilon,
        struct grid * dev_grid,
    	unsigned int * dev_indexLookupArr,
        struct gridCellLookup * dev_gridCellLookupArr,
        DTYPE * dev_minArr,
    	unsigned int * dev_nCells,
        unsigned int * dev_nNonEmptyCells,
        unsigned int * retNumBatches,
        unsigned int * retGPUBufferSize);

unsigned long long GPUBatchEst_v2(
        unsigned int * DBSIZE,
        DTYPE * dev_database,
        unsigned int * dev_originPointIndex,
        DTYPE * dev_epsilon,
        struct grid * dev_grid,
        unsigned int * dev_indexLookupArr,
        struct gridCellLookup * gridCellLookupArr,
        DTYPE * dev_minArr,
        unsigned int * dev_nCells,
        unsigned int * dev_nNonEmptyCells,
        unsigned int * retNumBatches,
        unsigned int * retGPUBufferSize,
        std::vector< std::pair<unsigned int, unsigned int> > * batches);

unsigned long long callGPUBatchEstTest(
        unsigned int * DBSIZE,
        unsigned int sampleBegin,
        unsigned int sampleEnd,
        DTYPE * dev_database,
        DTYPE * dev_sortedDatabase,
        unsigned int * dev_originPointIndex,
        DTYPE * dev_epsilon,
        struct grid * dev_grid,
    	unsigned int * dev_indexLookupArr,
        struct gridCellLookup * dev_gridCellLookupArr,
        DTYPE * dev_minArr,
    	unsigned int * dev_nCells,
        unsigned int * dev_nNonEmptyCells,
        unsigned int * retNumBatches,
        unsigned int * retGPUBufferSize);

void constructNeighborTableKeyValueWithPtrs(
        int * pointIDKey,
        int * pointInDistValue,
        struct neighborTableLookup * neighborTable,
        int * pointersToNeighbors,
        unsigned int * cnt);

void warmUpGPU();

#endif

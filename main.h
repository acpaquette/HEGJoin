#ifndef MAIN_H
#define MAIN_H

#include <vector>
#include <cstdint>

#include "params.h"

void importNDDataset(
        std::vector<std::vector <DTYPE> > * dataPoints,
        char * fname);

void sortInNDBins(
        std::vector<std::vector <DTYPE> > * dataPoints);

void generateNDGridDimensions(
        std::vector<std::vector <DTYPE> > * NDdataPoints,
        DTYPE epsilon,
        DTYPE * minArr,
        DTYPE * maxArr,
        unsigned int * nCells,
        uint64_t * totalCells);

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
        unsigned int * nNDMaskElems);

uint64_t getLinearID_nDimensions(
        unsigned int * indexes,
        unsigned int * dimLen,
        unsigned int nDimensions);

void printNeighborTable(
    struct neighborTableLookup * neighborTable,
    unsigned int size);



#endif

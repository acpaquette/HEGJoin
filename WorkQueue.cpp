#include <utility>
#include <stdio.h>
#include <vector>
#include <iostream>

#include "WorkQueue.h"
#include "params.h"

template <typename T>
inline T min(T a, T b)
{
    return (a < b) ? a : b;
}

template <typename T>
inline T max(T a, T b)
{
    return (a > b) ? a : b;
}

unsigned int queueIndex = 1;
unsigned int queueIndexCPU = 1;
unsigned int staticQueryPoint = 0;

unsigned int maxNeighbors = 0;

bool gpuOffset = false;

// Useless if the CPU does not EGO-Sort while the GPU sorts by workload.
bool isQueueReady = false;

std::pair<unsigned int, unsigned int> getBatchFromQueue(
    unsigned int DBSIZE,
    unsigned int batchSize)
{
    unsigned int begin, end;
    if(gpuOffset)
    {
        #pragma omp critical
        {
            if(queueIndex < DBSIZE && queueIndex < queueIndexCPU && queueIndex != queueIndexCPU) {
                begin = queueIndex;
                end = min(begin + batchSize, queueIndexCPU);
                queueIndex = end;
            } 
            else {
                begin = 0;
                end = 0;
                queueIndex = DBSIZE;
            }
        }
    } 
    else {
        begin = 1;
        end = 0;
    }
    return std::make_pair(begin, end);
}

unsigned int gpuBatch = 0;
std::pair<unsigned int, unsigned int> getBatchFromQueue_v2(
    std::vector< std::pair<unsigned int, unsigned int> > batches)
{
    unsigned int begin, end;
    if(gpuOffset)
    {
        #pragma omp critical
        {

            if(queueIndex < queueIndexCPU && queueIndex != queueIndexCPU) {
                begin = batches[gpuBatch].first;
                end = min(batches[gpuBatch].second, queueIndexCPU);
                queueIndex = end;
                gpuBatch++;
            }
            else {
                begin = 0;
                end = 0;
            }
        }
    }
    else {
        begin = 1;
        end = 0;
    }
    return std::make_pair(begin, end);
}

std::pair<unsigned int, unsigned int> getBatchFromQueueCPU(
    unsigned int DBSIZE,
    unsigned int batchSize)
{
    unsigned int begin, end;
    if(gpuOffset)
    {
        #pragma omp critical
        {
            if(0 < queueIndexCPU && queueIndex < queueIndexCPU && queueIndex != queueIndexCPU) {
                begin = max(queueIndex, queueIndexCPU - batchSize);
                end = queueIndexCPU;
                queueIndexCPU = begin;
            }
            else {
                begin = 0;
                end = 0;
            }
        }
    }
    else {
        begin = 1;
        end = 0;
    }
    return std::make_pair(begin, end);
}

void setQueueIndex(unsigned int index)
{
    queueIndex = index;
    gpuOffset = true;
}

void setQueueIndexCPU(unsigned int index)
{
    queueIndexCPU = index;
}

void setStaticQueryPoint(unsigned int index)
{
    staticQueryPoint = index;
}

unsigned int getStaticQueryPoint()
{
    return staticQueryPoint;
}

void displayIndexes()
{
    printf("[QUEUE] ~ GPU index: %d, CPU index: %d, GPU batch: %d\n", queueIndex, queueIndexCPU, gpuBatch);
}

void setWorkQueueReady()
{
    isQueueReady = true;
}

bool getWorkQueueReady()
{
    return isQueueReady;
}

void setMaxNeighbors(unsigned int n)
{
    maxNeighbors = n;
}

unsigned int getMaxNeighbors()
{
    return maxNeighbors;
}

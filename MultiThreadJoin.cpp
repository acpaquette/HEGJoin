//Copyright (c) 2013, Dmitri V. Kalashnikov. All rights reserved.
//This copyright notice should remain at the top of this file.
//

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <cstdint>
#include "omp.h"
#include <iostream>

#include "Point.hpp"
#include "Util.hpp"

#include "params.h"
#include "WorkQueue.h"
#include "structs.h"

using namespace std;

uint64_t Util::multiThreadJoinWorkQueue(
	unsigned int searchMode,
	pPoint A, int A_sz,
	pPoint B, int B_sz,
	unsigned int * egoMapping,
	unsigned int * originPointIndex,
	neighborTableLookup * neighborTable,
	uint64_t * nbCandidatesArray)
{
	double tStart = omp_get_wtime();

	uint64_t * results = new uint64_t[CPU_THREADS];
	unsigned int * nbQueries = new unsigned int[CPU_THREADS];
	uint64_t * nbCandidates = new uint64_t[CPU_THREADS];
	for (unsigned int i = 0; i < CPU_THREADS; ++i)
	{
		results[i] = 0;
		nbQueries[i] = 0;
		nbCandidates[i] = 0;
	}

	if (searchMode == SM_HYBRID)
	{
		#pragma omp parallel num_threads(CPU_THREADS)
		{
			unsigned int tid = omp_get_thread_num();
			// std::list<int> resultVector;
			std::pair<unsigned int, unsigned int> cpuBatch;
			// Point * batch = new Point[CPU_BATCH_SIZE];

			// unsigned int * tmpBuffer = new unsigned int[getMaxNeighbors()];
			// unsigned int * nbNeighbors = new unsigned int;

			do
			{
				cpuBatch = getBatchFromQueueCPU(A_sz, CPU_BATCH_SIZE);
			}while(cpuBatch.second < cpuBatch.first);

			do
			{
				// printf("[EGO | T_%d] ~ Begin: %d, end: %d\n", tid, cpuBatch.first, cpuBatch.second);
				nbQueries[tid] += cpuBatch.second - cpuBatch.first;
				unsigned int indexmaxPrec = 0;
				std::vector<int> * resultVector = new std::vector<int>();

				for (unsigned int i = cpuBatch.first; i < cpuBatch.second; ++i)
				{
					// (*nbNeighbors) = 0;
					#if SORT_BY_WORKLOAD
						unsigned int index = egoMapping[ originPointIndex[i] ];
					#else
						unsigned int index = egoMapping[i];
					#endif

					Util::egoJoinV2(A, 0, A_sz - 1, B, index, index, 0, &nbCandidates[tid], resultVector, nbCandidatesArray);
					// Util::egoJoinV2(A, 0, A_sz - 1, B, index, index, 0, tmpBuffer, nbNeighbors);

					// unsigned int tmpIndex = originPointIndex[i];
					// neighborTable[tmpIndex].pointID = tmpIndex;
					// neighborTable[tmpIndex].indexmin = 0;
					// neighborTable[tmpIndex].indexmax = (*nbNeighbors) - 1;
					// neighborTable[tmpIndex].dataPtr = new int[(*nbNeighbors)];
					// std::copy(tmpBuffer, tmpBuffer + (*nbNeighbors), neighborTable[tmpIndex].dataPtr);
					// results[tid] += (*nbNeighbors);

					unsigned int tmpIndex = originPointIndex[i];
					neighborTable[tmpIndex].pointID = tmpIndex;
					neighborTable[tmpIndex].indexmin = indexmaxPrec;
					neighborTable[tmpIndex].indexmax = resultVector->size();
					indexmaxPrec = resultVector->size();
				}

				for (unsigned int i = cpuBatch.first; i < cpuBatch.second; ++i)
				{
					unsigned int index = originPointIndex[i];
					// pPoint tmpPoint = &B[index];
					neighborTable[index].dataPtr = resultVector->data();
				}

				resultVector->shrink_to_fit();
				results[tid] += resultVector->size();
				delete resultVector;

				cpuBatch = getBatchFromQueueCPU(A_sz, CPU_BATCH_SIZE);
			}
			while (0 != cpuBatch.second);

			// results[tid] += resultVector.size() / 2;

			// resultVector.clear();
			// resultVector.shrink_to_fit();

			// delete[] tmpBuffer;
		}
	}
	else // only use the CPU, not the GPU
	{
		#pragma omp parallel num_threads(CPU_THREADS)
		{
			unsigned int tid = omp_get_thread_num();
			// std::vector<int> resultVector;
			std::pair<unsigned int, unsigned int> cpuBatch;

			// unsigned int * tmpBuffer = new unsigned int[getMaxNeighbors()];
			// unsigned int * nbNeighbors = new unsigned int;

			if (searchMode == SM_HYBRID_STATIC)
			{
				do
				{
					cpuBatch = getBatchFromQueueCPU(A_sz, CPU_BATCH_SIZE);
				}
				while (cpuBatch.second < cpuBatch.first);
			}
			else {
				cpuBatch = getBatchFromQueue(A_sz, CPU_BATCH_SIZE);
			}

			do
			{
				nbQueries[tid] += cpuBatch.second - cpuBatch.first;

				unsigned int indexmaxPrec = 0;
				// std::vector<int> * resultVector = new std::vector<int>(__max(getMaxNeighbors() / (cpuBatch.first + 1), CPU_BATCH_SIZE));
				std::vector<int> * resultVector = new std::vector<int>();
				for (unsigned int i = cpuBatch.first; i < cpuBatch.second; ++i)
				{
					// (*nbNeighbors) = 0;
					#if SORT_BY_WORKLOAD
						unsigned int index = egoMapping[ originPointIndex[i] ];
					#else
						unsigned int index = egoMapping[i];
					#endif
					Util::egoJoinV2(A, 0, A_sz - 1, B, index, index, 0, &nbCandidates[tid], resultVector, nbCandidatesArray);
					// Util::egoJoinV2(A, 0, A_sz - 1, B, index, index, 0, tmpBuffer, nbNeighbors);

					// unsigned int tmpIndex = originPointIndex[i];
					// neighborTable[tmpIndex].pointID = tmpIndex;
					// neighborTable[tmpIndex].indexmin = 0;
					// neighborTable[tmpIndex].indexmax = (*nbNeighbors) - 1;
					// neighborTable[tmpIndex].dataPtr = new int[(*nbNeighbors)];
					// std::copy(tmpBuffer, tmpBuffer + (*nbNeighbors), neighborTable[tmpIndex].dataPtr);
					// results[tid] += (*nbNeighbors);

					#if SORT_BY_WORKLOAD
						unsigned int tmpIndex = originPointIndex[i];
					#else
						unsigned int tmpIndex = i;
					#endif
					neighborTable[tmpIndex].pointID = tmpIndex;
					neighborTable[tmpIndex].indexmin = indexmaxPrec;
					neighborTable[tmpIndex].indexmax = resultVector->size();
					indexmaxPrec = resultVector->size();
				}

				for (unsigned int i = cpuBatch.first; i < cpuBatch.second; ++i)
				{
					#if SORT_BY_WORKLOAD
						unsigned int tmpIndex = originPointIndex[i];
					#else
						unsigned int tmpIndex = i;
					#endif
					neighborTable[tmpIndex].dataPtr = resultVector->data();
				}
				resultVector->shrink_to_fit();
				results[tid] += resultVector->size();
				delete resultVector;

				if (searchMode == SM_HYBRID_STATIC)
				{
					cpuBatch = getBatchFromQueueCPU(A_sz, CPU_BATCH_SIZE);
				} else {
					cpuBatch = getBatchFromQueue(A_sz, CPU_BATCH_SIZE);
				}
			}while (0 != cpuBatch.second);

			// results[tid] += resultVector.size() / 2;
			// results[tid] += resultVector.size();

			// resultVector.clear();
			// resultVector.shrink_to_fit();

			// delete[] tmpBuffer;
		}
	}
	double tEnd = omp_get_wtime();

	uint64_t result = 0;
	unsigned int nbQueriesTotal = 0;
	uint64_t nbCandidatesTotal = 0;
	for(int i = 0; i < CPU_THREADS; ++i)
	{
		result += results[i];
		nbQueriesTotal += nbQueries[i];
		nbCandidatesTotal += nbCandidates[i];
	}

	printf("[EGO | RESULT] ~ Query points computed by Super-EGO: %d\n", nbQueriesTotal);
	printf("[EGO | RESULT] ~ Candidate points refined by Super-EGPU: %llu\n", nbCandidatesTotal);
	printf("[EGO | RESULT] ~ Compute time for Super-EGO: %f\n", tEnd - tStart);

	delete[] results;
	delete[] nbQueries;
	delete[] nbCandidates;

	return result;
}


// uint64_t Util::multiThreadJoinPreQueue(
// 	pPoint A, int A_sz,
// 	pPoint B, int B_sz,
// 	unsigned int * egoMapping,
// 	struct grid * grid,
// 	unsigned int * indexLookupArr,
// 	struct gridCellLookup * gridCellLookupArr,
// 	unsigned int * nNonEmptyCells,
// 	bool * isSortByWLDone,
// 	unsigned int * nbPointsComputedReturn,
// 	CPU_State * cpuState,
// 	neighborTableLookup * neighborTable)
// {
//
// 	double tStart = omp_get_wtime();
//
// 	uint64_t * results = new uint64_t[CPU_THREADS];
// 	unsigned int * nbQueries = new unsigned int[CPU_THREADS];
// 	for(int i = 0; i < CPU_THREADS; ++i)
// 	{
// 		results[i] = 0;
// 		nbQueries[i] = 0;
// 	}
//
// 	#pragma omp critical
// 	{
// 		if(!(*isSortByWLDone))
// 		{
// 			(*cpuState) = CPU_State::computing;
// 			printf("[EGO pre-Q] ~ Starting computation\n");
// 		}
// 	}
//
// 	unsigned int nbPointsComputed = 0;
// 	#pragma omp parallel num_threads(CPU_THREADS)
// 	{
// 		unsigned int tid = omp_get_thread_num();
//
// 		bool localSortByWLDone = false;
// 		unsigned int localNbPointsComputed = 0;
// 		// std::vector<unsigned int> neighborsList;
//
// 		unsigned int tmpIndex, index;
//
// 		while(!localSortByWLDone)
// 		{
// 			// Eventually change this to compute several points to benefit from cache
// 			#pragma omp critical
// 			{
// 				localNbPointsComputed = nbPointsComputed;
// 				nbPointsComputed += CPU_BATCH_SIZE;
// 			}
//
// 			// tmpIndex = indexLookupArr[localNbPointsComputed];
// 			// pPoint tmpPoint = &B[index];
// 			// printf("nbPointsComputed = %d, index = %d, point id = %d\n", localNbPointsComputed, index, tmpPoint->id);
//
// 			std::vector<int> * neighborList = new std::vector<int>();
// 			unsigned int indexmaxPrec = 0;
//
// 			for(int i = localNbPointsComputed; i < localNbPointsComputed + CPU_BATCH_SIZE; ++i)
// 			{
// 				index = egoMapping[i];
//
// 				Util::egoJoinV2(A, 0, A_sz - 1, B, index, index, 0, &nbCandidate[i], neighborList);
//
// 				neighborTable[i].pointID = i;
// 				neighborTable[i].indexmin = indexmaxPrec;
// 				neighborTable[i].indexmax = neighborList->size();
// 				indexmaxPrec = neighborList->size();
// 			}
//
// 			for(int i = localNbPointsComputed; i < localNbPointsComputed + CPU_BATCH_SIZE; ++i)
// 			{
// 				neighborTable[i].dataPtr = neighborList->data();
// 			}
//
// 			// neighborTable[localNbPointsComputed].dataPtr = neighborList->data();
//
// 			neighborList->shrink_to_fit();
// 			results[tid] += neighborList->size();
//
// 			#pragma omp critical
// 			{
// 				localSortByWLDone = (*isSortByWLDone);
// 			}
// 		}
// 	}
//
// 	double tEnd = omp_get_wtime();
//
// 	#pragma omp critical
// 	{
// 		(*nbPointsComputedReturn) = nbPointsComputed;
// 		(*cpuState) = CPU_State::doneComputing;
// 	}
//
// 	uint64_t resultTotal = 0;
// 	// unsigned int nbQueriesTotal = 0;
// 	unsigned int nbQueriesTotal = nbPointsComputed;
// 	for(int i = 0; i < CPU_THREADS; ++i)
// 	{
// 		resultTotal += results[i];
// 		// nbQueriesTotal += nbQueries[i];
// 	}
//
// 	printf("[EGO pre-Q | RESULT] ~ Query points computed by Super-EGO while sorting by workload was running: %d\n", nbQueriesTotal);
// 	printf("[EGO pre-Q | RESULT] ~ Compute time for Super-EGO while sorting by workload was running: %f\n", tEnd - tStart);
//
// 	delete[] results;
// 	delete[] nbQueries;
//
// 	return resultTotal;
//
//
// }

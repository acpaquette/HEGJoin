//Copyright (c) 2013, Dmitri V. Kalashnikov. All rights reserved.
//This copyright notice should remain at the top of this file.
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cstdint>
#include <list>
#include "omp.h"
#include <utility>
#include <cmath>
#include <iostream>

#include "Point.hpp"
#include "Util.hpp"

#include "params.h"

using namespace std;

REAL Util::eps;
REAL Util::eps2; //eps squared

int Util::r1[GPUNUMDIM + 1][2]; //ranges for SimpleJoin
int Util::r2[GPUNUMDIM + 1][2];
int Util::r3[GPUNUMDIM + 1][2];

static double seed = 1234567.0; // init only one time

//-----------------------------------------
void Util::reorderDim(pPoint A, int A_sz, pPoint B, int B_sz)
{
    //-- define stat vars --
    int num_buck = ceil(1 / eps) + 1;   //number of buckets
	// int num_buck = 1201;
    pPoint hA = new Point[num_buck];    //histogram for A
    pPoint hB = new Point[num_buck];    //histogram for B
    double d[GPUNUMDIM];                  //avg distance in each dim

    //-- init stats to zeroes --
    for(int i = 0; i < GPUNUMDIM; i++)
        d[i] = 0;

    for (int i = 0; i < num_buck; i++)
    {
        for (int j = 0; j < GPUNUMDIM; j++)
        {
            hA[i].x[j] = 0;
            hB[i].x[j] = 0;
        }
    }

    //-- sample points --
    int sample_sz = 2000;

    for (int i = 0; i < sample_sz; i++)
    {
        //-- get rnd point from A --
        int rnd_i = (A_sz - 1) * rnd();
        pPoint rnd_pA = &A[rnd_i];

        //-- get rnd point from B --
        rnd_i = (B_sz - 1) * rnd();
        pPoint rnd_pB = &B[rnd_i];

        //-- update stats --
        for (int j = 0; j < GPUNUMDIM; j++)
        {
            d[j] += fabs(rnd_pA->x[j] - rnd_pB->x[j]);

            int buck_i = (int) (rnd_pA->x[j] / eps);
            hA[buck_i].x[j] += 1;

            buck_i = (int) (rnd_pB->x[j] / eps);
            hB[buck_i].x[j] += 1;
        }
    }

    //-- compute fail factor f in each dimension --
    long double f[GPUNUMDIM];
    long double g[GPUNUMDIM];

    for (int i = 0; i < GPUNUMDIM; i++)
    {
        f[i] = 0;

        for (int j = 0; j < num_buck; j++)
        {
            if (j == 0)
            {
                f[i] += hA[j].x[i] * (hB[j].x[i] + hB[j+1].x[i]);
                continue;
            }
            else if (j == (num_buck - 1) )
            {
                f[i] += hA[j].x[i] * (hB[j].x[i] + hB[j-1].x[i]);
                continue;
            }

            f[i] += hA[j].x[i] * (hB[j].x[i] + hB[j-1].x[i] + hB[j+1].x[i]);
        }

    }

    delete[] hA;
    delete[] hB;

    //-- normalizing f --
    for (int i = 0; i < GPUNUMDIM; i++)
    {
        d[i] = d[i] / sample_sz;
        f[i] = f[i] / (sample_sz * sample_sz);
        g[i] = -d[i]; //f[i];
        //printf("\n f[%d] = %f", i, f[i]);
    }

    //-- constracting map of remapping (not efficient) --
    int map[GPUNUMDIM];

    for (int i = 0; i < GPUNUMDIM; i++)
    {
        double min = 1000;
        int min_j = -1;

        for (int j = 0; j < GPUNUMDIM; j++)
        {
            if (g[j] < min)
            {
                min = g[j];
                min_j = j;
            }
        }

        map[i] = min_j;
        g[min_j] = 2.0; // so that it is not min next time
    }

    printf("\n\n -- map --");
    for (int i = 0; i < GPUNUMDIM; i++)
    {
        printf("\n s[%2d] = %.3f, d[%2d] = %.3f, map[%2d] = %2d", i, 1 - f[i], i, d[i], i, map[i]);
    }
    printf("\n-- --- --\n\n");

    //-- reoder dimension in A --
    for (int i = 0; i < A_sz; i++)
    {
        REAL x[GPUNUMDIM];

        for (int j = 0; j < GPUNUMDIM; j++)
            x[j] = A[i].x[j];

        for (int j = 0; j < GPUNUMDIM; j++)
            A[i].x[j] = x[map[j]];
    }

    //-- reoder dimension in B, if not self-join --
    if(B != A)
    {
        for (int i = 0; i < B_sz; i++)
        {
            REAL x[GPUNUMDIM];


            for (int j = 0; j < GPUNUMDIM; j++)
                x[j] = B[i].x[j];

            for (int j = 0; j < GPUNUMDIM; j++)
                B[i].x[j] = x[map[j]];
        }
    }

    //-- reorder stats accrodingly as well --
    double rs[GPUNUMDIM];
    double rd[GPUNUMDIM];

    for (int j = 0; j < GPUNUMDIM; j++)
    {
        rs[j] = 1 - f[map[j]];
        rd[j] = d[map[j]];
    }

    printf("\n\n -- remap --");
    for (int i = 0; i < GPUNUMDIM; i++)
    {
        printf("\n rs[%2d] = %.3f, rd[%2d] = %.3f", i, rs[i], i, rd[i]);
    }
    printf("\n-- --- --\n\n");

    //-- Case 1: zero inactive dimensions --
    int sml_seq_sz = __min(1, GPUNUMDIM);

    for (int i = 0; i < sml_seq_sz; i++)
    {
        r1[i][0] = 0;
        r1[i][1] = GPUNUMDIM-1;

        r2[i][0] = 0;
        r2[i][1] = -1;

        r3[i][0] = 0;
        r3[i][1] = -1;
    }

    //-- Case 2: all dims are inactive --
    r1[GPUNUMDIM][0] = 0;
    r1[GPUNUMDIM][1] = GPUNUMDIM-1;

    r2[GPUNUMDIM][0] = 0;
    r2[GPUNUMDIM][1] = -1;

    r3[GPUNUMDIM][0] = 0;
    r3[GPUNUMDIM][1] = -1;

    //-- Case 3: remaining cases --
    //-- find first k s.t. rd[k] < eps/2 --
    int k = GPUNUMDIM;

    for (int i = 0; i < GPUNUMDIM; i++)
    {
        if (rd[i] < eps/2)
        {
            k = i;
            break;
        }
    }

    //--
    for (int i = sml_seq_sz; i < GPUNUMDIM; i++)
    {
        //-- Case I: 1-interval --
        if (rd[i] < eps/2)
        {
            r1[i][0] = 0;
            r1[i][1] = GPUNUMDIM-1;

            r2[i][0] = 0;
            r2[i][1] = -1;

            r3[i][0] = 0;
            r3[i][1] = -1;

            continue;
        }

        //-- Case II: 3-intervals --
        if (k < GPUNUMDIM)
        {
            r1[i][0] = i;
            r1[i][1] = k - 1;

            r2[i][0] = 0;
            r2[i][1] = i - 1;

            r3[i][0] = k;
            r3[i][1] = GPUNUMDIM-1;

            continue;
        }

        //-- Case III: 2-interval --
        r1[i][0] = i; //in simj, start by scanning first unproc dim
        r1[i][1] = GPUNUMDIM-1;

        r2[i][0] = 0;
        r2[i][1] = i - 1;

        r3[i][0] = 0;
        r3[i][1] = -1;
    }

    //-- print scan table --
    printf("\n--- scan table --");
    for (int i = 0; i <= GPUNUMDIM; i++)
    {
        printf("\n  d=%2d: r1=[%2d,%2d] r2=[%2d,%2d] r3=[%2d,%2d]", i, r1[i][0], r1[i][1], r2[i][0], r2[i][1], r3[i][0], r3[i][1]);
    }
    printf("\n--- ------ --\n\n");
}


//-----------------------------------------
int pcmp(const void *v1, const void *v2)
{
	pPoint p1 = (pPoint)v1;
	pPoint p2 = (pPoint)v2;

	for (int i = 0; i < GPUNUMDIM; i++)
	{
		int d = ((int) (p1->x[i]/Util::eps)) - ((int) (p2->x[i]/Util::eps));

		if (d != 0)
			return d;
	}

	return 0;
	// for (int i = 0; i < GPUNUMDIM; i++)
	// {
	// 	if ( (int) (p1->x[i] / Util::eps) < (int)(p2->x[i] / Util::eps) ) return true;
	// 	if ( (int) (p1->x[i] / Util::eps) > (int)(p2->x[i] / Util::eps) ) return false;
	// }
	//
	// return false;
}



////////////////////////////////////////////////////////////////////////////////



void Util::egoJoinV2(pPoint A, int frA, int toA, pPoint B, int frB, int toB, int start_dim, uint64_t * nbCandidate, std::vector<int> * result, uint64_t * nbCandidatesArray)
// void Util::egoJoinV2(pPoint A, int frA, int toA, pPoint B, int frB, int toB, int start_dim, unsigned int * result, unsigned int * nbNeighbors)
{
	pPoint fst_A = &A[frA];
	pPoint lst_A = &A[toA];
	pPoint fst_B = &B[frB];
	pPoint lst_B = &B[toB];

	int i;
	for (i = start_dim; i < GPUNUMDIM; i++)
	{
		int lo_A = (int) (fst_A->x[i] / eps);
		int hi_B = (int) (lst_B->x[i] / eps);

		if (lo_A > hi_B + 1) // lo_B--------hi_B  lo_A--------hi_A
			return;

		int lo_B = (int) (fst_B->x[i] / eps);
		int hi_A = (int) (lst_A->x[i] / eps);

		if (lo_B > hi_A + 1) // lo_A--------hi_A  lo_B---------hi_B
			return;

		if ( (lo_A < hi_A) || (lo_B < hi_B) ) // the remaining dimensions will always intersect
		{
			start_dim = i;
			break;
		}
	}

	int A_sz = toA - frA + 1;
	int B_sz = toB - frB + 1;

	if (A_sz < MINLEN && B_sz < MINLEN)
	{
        simpleJoin4(A, frA, toA, B, frB, toB, i, nbCandidate, result, nbCandidatesArray);
		// simpleJoin4(A, frA, toA, B, frB, toB, i, result, nbNeighbors);

		return;
	}

	if (A_sz >= MINLEN && B_sz >= MINLEN)
	{
        // printf("\nCase 2 (self-thread)\n");
        egoJoinV2(A, frA             , frA + A_sz / 2, B, frB             , frB + B_sz/2, start_dim, nbCandidate, result, nbCandidatesArray); // f f
		egoJoinV2(A, frA             , frA + A_sz / 2, B, frB + B_sz/2 + 1, toB         , start_dim, nbCandidate, result, nbCandidatesArray); // f s
		egoJoinV2(A, frA + A_sz/2 + 1, toA           , B, frB             , frB + B_sz/2, start_dim, nbCandidate, result, nbCandidatesArray); // s f
		egoJoinV2(A, frA + A_sz/2 + 1, toA           , B, frB + B_sz/2 + 1, toB         , start_dim, nbCandidate, result, nbCandidatesArray); // f s
		// egoJoinV2(A, frA             , frA + A_sz / 2, B, frB             , frB + B_sz/2, start_dim, result, nbNeighbors); // f f
		// egoJoinV2(A, frA             , frA + A_sz / 2, B, frB + B_sz/2 + 1, toB         , start_dim, result, nbNeighbors); // f s
		// egoJoinV2(A, frA + A_sz/2 + 1, toA           , B, frB             , frB + B_sz/2, start_dim, result, nbNeighbors); // s f
		// egoJoinV2(A, frA + A_sz/2 + 1, toA           , B, frB + B_sz/2 + 1, toB         , start_dim, result, nbNeighbors); // f s
		return;
	}

	//-- case 3--
	if (A_sz >= MINLEN && B_sz < MINLEN)
	{
        // printf("\nCase 3 (self-thread)\n");
        egoJoinV2(A, frA             , frA + A_sz / 2, B, frB, toB, start_dim, nbCandidate, result, nbCandidatesArray); // f full
		egoJoinV2(A, frA + A_sz/2 + 1, toA           , B, frB, toB, start_dim, nbCandidate, result, nbCandidatesArray); // s full
		// egoJoinV2(A, frA             , frA + A_sz / 2, B, frB, toB, start_dim, result, nbNeighbors); // f full
		// egoJoinV2(A, frA + A_sz/2 + 1, toA           , B, frB, toB, start_dim, result, nbNeighbors); // s full
		return;
	}

	//-- case 4 --
	if (A_sz < MINLEN && B_sz >= MINLEN)
	{
        // printf("\nCase 4 (self-thread)\n");
        egoJoinV2(A, frA, toA, B, frB             , frB + B_sz/2, start_dim, nbCandidate, result, nbCandidatesArray); // f f
		egoJoinV2(A, frA, toA, B, frB + B_sz/2 + 1, toB         , start_dim, nbCandidate, result, nbCandidatesArray); // f s
		// egoJoinV2(A, frA, toA, B, frB             , frB + B_sz/2, start_dim, result, nbNeighbors); // f f
		// egoJoinV2(A, frA, toA, B, frB + B_sz/2 + 1, toB         , start_dim, result, nbNeighbors); // f s
		return;
	}
}



void Util::simpleJoin3(pPoint A, int frA, int toA, pPoint B, int frB, int toB, uint64_t * nbCandidate, std::vector<int> * result, uint64_t * nbCandidatesArray)
// void Util::simpleJoin3(pPoint A, int frA, int toA, pPoint B, int frB, int toB, unsigned int * result, unsigned int * nbNeighbors)
{
	// for (int i = frA; i <= toA; i++)
	for(int i = frB; i <= toB; ++i)
	{
		// pPoint p = &A[i];
		pPoint q = &B[i];

        #if COUNT_CANDIDATES
            (*nbCandidate) += (toA - frA);
        #endif

        #if COMPARE_CANDIDATES
            nbCandidatesArray[q->id] += (toA - frA);
        #endif

		// for (int j = frB; j <= toB; j++)
		for(int j = frA; j <= toA; ++j)
		{
			// pPoint q = &B[j];
			pPoint p = &A[j];

			REAL sum = 0;

			for (int k = 0; k < GPUNUMDIM; k++)
			{
				REAL dx = (p->x[k] - q->x[k]);
				dx = dx * dx;
				sum += dx;

				if (sum > eps2)
					goto stop1;
			}

			result->push_back(p->id);
			// result->push_back(q->id);
            // result[(*nbNeighbors)] = p->id;
            // (*nbNeighbors) += 1;

			stop1: ;
		}
	}

	return;
}



void Util::simpleJoin4(pPoint A, int frA, int toA, pPoint B, int frB, int toB, int m, uint64_t * nbCandidate, std::vector<int> * result, uint64_t * nbCandidatesArray)
// void Util::simpleJoin4(pPoint A, int frA, int toA, pPoint B, int frB, int toB, int m, unsigned int * result, unsigned int * nbNeighbors)
{
	int r1_beg = r1[m][0];
    int r1_end = r1[m][1];

    int r2_end = r2[m][1];

    int r3_beg = r3[m][0];
    int r3_end = r3[m][1];

	if (r3_end == -1) // 1 or 2 intervals
    {
        if (r2_end == -1) // 1 interval
        {
            simpleJoin3(A, frA, toA, B, frB, toB, nbCandidate, result, nbCandidatesArray);
            // simpleJoin3(A, frA, toA, B, frB, toB, result, nbNeighbors);
            return;
        }

        //-- 2 intervals --
        // for (int i = frA; i <= toA; i++)
		for(int i = frB; i <= toB; ++i)
        {
            // pPoint p = &A[i];
			pPoint q = &B[i];

            #if COUNT_CANDIDATES
                (*nbCandidate) += (toA - frA);
            #endif

            #if COMPARE_CANDIDATES
                nbCandidatesArray[q->id] += (toA - frA);
            #endif

            // for (int j = frB; j <= toB; j++)
			for(int j = frA; j <= toA; ++j)
            {
                // pPoint q = &B[j];
				pPoint p = &A[j];

                REAL sum = 0;

                //-- scan over range 1 --
                for (int k = r1_beg; k <= GPUNUMDIM - 1; k++)
                {
                    REAL dx = (p->x[k] - q->x[k]);
                    dx = dx * dx;
                    sum += dx;

                    if (sum > eps2)
                        goto stop_2_int;
                }

                //-- scan over range 2 --
                for (int k = 0; k <= r2_end; k++)
                {
                    REAL dx = (p->x[k] - q->x[k]);
                    dx = dx * dx;
                    sum += dx;

                    if (sum > eps2)
                        goto stop_2_int;
                }

                result->push_back(p->id);
                // result->push_back(q->id);
                // result[(*nbNeighbors)] = p->id;
                // (*nbNeighbors) += 1;

            stop_2_int: ;

            }
        }

        return;
    }


	//-- 3 intervals --
	// for (int i = frA; i <= toA; i++)
	for(int i = frB; i <= toB; ++i)
	{
		// pPoint p = &A[i];
		pPoint q = &B[i];

        #if COUNT_CANDIDATES
            (*nbCandidate) += (toA - frA);
        #endif

        #if COMPARE_CANDIDATES
            nbCandidatesArray[q->id] += (toA - frA);
        #endif

		// for (int j = frB; j <= toB; j++)
		for(int j = frA; j <= toA; ++j)
		{
			// pPoint q = &B[j];
			pPoint p = &A[j];

			REAL sum = 0;

            //-- scan over range 1 --
            for (int k = r1_beg; k <= r1_end; k++)
            {
                REAL dx = (p->x[k] - q->x[k]);
                dx = dx * dx;
                sum += dx;

                if (sum > eps2)
                    goto stop2;
            }

            //-- scan over range 2 --
            for (int k = 0; k <= r2_end; k++)
            {
                REAL dx = (p->x[k] - q->x[k]);
                dx = dx * dx;
                sum += dx;

                if (sum > eps2)
                    goto stop2;
            }

            //-- scan over range 3 --
            for (int k = r3_beg; k <= GPUNUMDIM - 1; k++)
            {
                REAL dx = (p->x[k] - q->x[k]);
                dx = dx * dx;
                sum += dx;

                if (sum > eps2)
                    goto stop2;
            }

			result->push_back(p->id);
			// result->push_back(q->id);
            // result[(*nbNeighbors)] = p->id;
            // (*nbNeighbors) += 1;

            stop2: ;

		}
	}

	return;
}



////////////////////////////////////////////////////////////////////////////////



REAL Util::rnd()
{
	const double d2p31m = 2147483647.0;
	const double d2p31  = 2147483711.0;

    long double temp1 = floor(16807 * seed / d2p31m) * d2p31m;
    long double temp2 = 16807 * seed;
    long double new_seed = temp2 - temp1;
    seed = new_seed;

    return (REAL)(fabs(seed / d2p31));
}

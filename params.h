#ifndef PARAMS_H
#define PARAMS_H

// Input data dimension
// #define GPUNUMDIM 90

// Number of dimensions to index
// #define NUMINDEXEDDIM 6

// Number of GPU threads per block
#define BLOCKSIZE 256
#define PBLOCKS 256

// Number of GPU streams
#define GPUSTREAMS 1

// Number of CPU threads to use when joining with the CPU
// #define CPU_THREADS 8

// Number of query points taken by a CPU thread at once
#define CPU_BATCH_SIZE 1024

// Note: Super-EGO does not work using double precision
// TODO should merge the two variables together
#define DTYPE double // precision used by the GPU component
#define REAL double  // precision used by Super-EGO

// Used by Super-EGO
#define MINLEN 32

// Statically split the work based on the queries (e.g., 60% of the queries to the GPU, 40% to the CPU)
// Or statically split the work based on the candidate points (e.g., 60% of the candidates to refine
//  go to the GPU and corresponding to 25% of the queries for example, and 40% of the candidates to refine
//  go to the CPU, corresponding to the other 75% of the queries for example)
#define STATIC_SPLIT_QUERIES 1

// Limits the output produced by the GPU when computing
// 0 to output everything, 1 to limit the output
#define SILENT_GPU 0

// Metrics to evaluate the throughput of the CPU and GPU by counting the number of candidate points they refine
#define COUNT_CANDIDATES 0
#define COUNT_CANDIDATES_GPU 0
#define COMPARE_CANDIDATES 0



/*********************************************************************/
/*                 Code below should not be modified                 */
/*********************************************************************/



#define NB_ARGS_MAX 6
#define FILENAME_ARG 1
#define EPSILON_ARG 2
#define DIM_ARG 3
#define SEARCHMODE_ARG 4
#define STATIC_PART_ARG 5

#define SM_GPU 0
#define SM_HYBRID 1
#define SM_HYBRID_STATIC 2
#define SM_CPU 3

// Sorts the point by workload
#define SORT_BY_WORKLOAD 1

#endif

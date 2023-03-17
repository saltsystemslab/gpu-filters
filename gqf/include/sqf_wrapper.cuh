/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *                  Hunter McCoy <hjmccoy@lbl.gov> 
 *
 * ============================================================================
 */

#ifndef sqf_WRAPPER_CUH
#define sqf_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "sqf.cuh"
#include <climits>

#include <chrono>
#include <iostream>

struct sqf_filter::quotient_filter sqf_cqf_gpu;

unsigned int * sqf_inserts;
unsigned int * sqf_returns;






#ifndef NUM_SLOTS_TO_LOCK
#define NUM_SLOTS_TO_LOCK (1ULL<<13)
#endif

extern inline uint64_t sqf_xnslots();

extern inline int sqf_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	cudaMalloc((void **)& sqf_inserts, sizeof(unsigned int)*buf_size);
	cudaMalloc((void **)& sqf_returns, sizeof(int)*buf_size);


	//__host__ void initCQFGPU(struct countingQuotientFilterGPU *cqf, unsigned int q);
	sqf_filter::initFilterGPU(&sqf_cqf_gpu, nbits, 5);
	//initCQFGPU(&test_cqf_gpu, nbits);

	return 0;
}

extern inline int sqf_insert(uint64_t val, uint64_t count)
{
	//qf_insert(g_quotient_filter, val, 0, count, QF_NO_LOCK);
	return 0;
}

extern inline int sqf_lookup(uint64_t val)
{

	return 0;
	//return qf_count_key_value(g_quotient_filter, val, 0, 0);
}

extern inline uint64_t sqf_range()
{
	//fix me im on device
	//have to deep copy

	
	return 0;
}

extern inline uint64_t sqf_xnslots()
{
	//fix me im on device
	//have to deep copy

	return 0;
}

extern inline int sqf_destroy()
{
	//since its a struct I don't think we do anything?
	//the orignal code has no ~Filter 
	//I'll write my own if its a problem - this is a memory leak but it may not matter :D

	return 0;
}

extern inline int sqf_iterator(uint64_t pos)
{
	//qf_iterator_from_position(g_quotient_filter, &g_quotient_filter_itr, pos);
	return 0;
}

/* Returns 0 if the iterator is still valid (i.e. has not reached the
 * end of the QF. */
extern inline int sqf_get(uint64_t *key, uint64_t *value, uint64_t *count)
{
	return 0; //qfi_get_hash(&g_quotient_filter_itr, key, value, count);
}

/* Advance to next entry.  Returns whether or not another entry is
 * found.  */
extern inline int sqf_next()
{
	return 0;
}

/* Check to see if the if the end of the QF */
extern inline int sqf_end()
{
	return 0;
}


__global__ void sqf_downcast(uint64_t nitems, uint64_t * src, unsigned int * dst){


	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nitems) return;


	dst[tid] = src[tid];


}

extern inline int sqf_bulk_insert(uint64_t * vals, uint64_t count)
{

	//calculate ratios
	//total_items += count;

	//int ratio = num_slots/total_items;

	//no sense in inflating the locks 200x for one insert
	//if (ratio > 15) ratio = 15;
	//printf("Dividing ratio %d\n", ratio);


	
	sqf_downcast<<<(count-1)/512+1, 512>>>(count, vals, sqf_inserts);


  //insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);

	//__host__ float insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);

	//will this work?
	//sqf_filter::bulkBuildSegmentedLayouts(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates);
	//sqf_filter::insert(sqf_cqf_gpu, count, sqf_inserts);
	//sqf_filter::bulkBuildSegmentedLayouts(sqf_cqf_gpu, count, sqf_inserts, false);
	
	//sqf_filter::bulkBuildParallelMerging(sqf_cqf_gpu, count, sqf_inserts, false);
	sqf_filter::insert(sqf_cqf_gpu, count, sqf_inserts);
	//insertGPU(sqf_cqf_gpu, count, sqf_inserts, sqf_returns);
	
  //cudaMemset((uint64_t *) buffer_sizes, 0, ratio*num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_one_hash(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK/ratio, num_locks*ratio, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
  //bulk_insert_bucketing_buffer_provided_timed(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_no_atomics(g_quotient_filter, vals,0,1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_sizes);

	cudaDeviceSynchronize();
	return 0;
}


__global__ void sqf_check(unsigned int * returns, uint64_t count, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= count) return;

	if (returns[tid] == UINT_MAX){
		atomicAdd((unsigned long long int *) misses, (unsigned long long int) 1);
	}

}

extern inline uint64_t sqf_bulk_get(uint64_t * vals, uint64_t count){


	// uint64_t * misses;
 //  cudaMallocManaged((void **)& misses, sizeof(uint64_t));
 //  misses[0] = 0;

  //auto start = std::chrono::high_resolution_clock::now();

  sqf_downcast<<<(count-1)/512+1, 512>>>(count, vals, sqf_inserts);
  //return bulk_get_wrapper(g_quotient_filter, vals, count);

  //cudaDeviceSynchronize();

  //auto midpoint = std::chrono::high_resolution_clock::now();

  sqf_filter::launchUnsortedLookups(sqf_cqf_gpu, count, sqf_inserts, sqf_returns);

  //launchLookups(test_cqf_gpu, count, sqf_inserts, sqf_returns);
  cudaDeviceSynchronize();

  //  auto end = std::chrono::high_resolution_clock::now();


  // //sqf_check<<<(count-1)/512+1, 512>>>(sqf_returns, count, misses);
  // cudaDeviceSynchronize();


  //std::cout << "downcast: " << (midpoint - start).count() << " rest." << (end - midpoint).count() << " \n";

  // uint64_t toReturn = misses[0];

  // cudaFree(misses);

  // return toReturn;

  return 0;


}


extern inline uint64_t sqf_bulk_get_fp(uint64_t * vals, uint64_t count){


  uint64_t * misses;
  cudaMallocManaged((void **)& misses, sizeof(uint64_t));
  misses[0] = 0;

  sqf_downcast<<<(count-1)/512+1, 512>>>(count, vals, sqf_inserts);
  //return bulk_get_wrapper(g_quotient_filter, vals, count);

  sqf_filter::launchUnsortedLookups(sqf_cqf_gpu, count, sqf_inserts, sqf_returns);

  //launchLookups(test_cqf_gpu, count, sqf_inserts, sqf_returns);
  cudaDeviceSynchronize();


  sqf_check<<<(count-1)/512+1, 512>>>(sqf_returns, count, misses);
  cudaDeviceSynchronize();

  uint64_t toReturn = misses[0];

  cudaFree(misses);

  return toReturn;

  //return 0;


}


//replace vals with a cudaMalloced Array for gpu inserts
extern inline uint64_t * sqf_prep_vals(__uint128_t * vals, uint64_t count){


	uint64_t *hostvals;
	//= (uint64_t * ) calloc(count, sizeof(uint64_t));
	cudaMallocManaged((void **)&hostvals, count*sizeof(uint64_t));

	for (uint64_t i=0; i < count; i++){
		hostvals[i] = vals[i];
	}

	return hostvals;
}


extern inline void sqf_bulk_delete(uint64_t * vals, uint64_t count){


	sqf_downcast<<<(count-1)/512+1, 512>>>(count, vals, sqf_inserts);


	superclusterDeletes(sqf_cqf_gpu, count, sqf_inserts);

	cudaDeviceSynchronize();
	return;
}


#endif

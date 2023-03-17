/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>  
 *                  Hunter McCoy <hjmccoy@lbl.gov>  
 *
 * ============================================================================
 */

#ifndef RSQF_WRAPPER_CUH
#define RSQF_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "RSQF.cuh"
#include <climits>

struct countingQuotientFilterGPU test_cqf_gpu;

unsigned int * rsqf_inserts;
int * rsqf_returns;




#ifndef NUM_SLOTS_TO_LOCK
#define NUM_SLOTS_TO_LOCK (1ULL<<13)
#endif


extern inline int rsqf_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	cudaMalloc((void **)& rsqf_inserts, sizeof(unsigned int)*buf_size);
	cudaMalloc((void **)& rsqf_returns, sizeof(int)*buf_size);


	//__host__ void initCQFGPU(struct countingQuotientFilterGPU *cqf, unsigned int q);
	initCQFGPU(&test_cqf_gpu, nbits);

	return 0;
}


extern inline int rsqf_destroy()
{
	//since its a struct I don't think we do anything?
	//the orignal code has no ~Filter 
	//I'll write my own if its a problem - this is a memory leak but it may not matter :D

	return 0;
}


__global__ void downcast(uint64_t nitems, uint64_t * src, unsigned int * dst){


	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nitems) return;


	dst[tid] = src[tid];


}

extern inline int rsqf_bulk_insert(uint64_t * vals, uint64_t count)
{

	//calculate ratios
	//total_items += count;

	//int ratio = num_slots/total_items;

	//no sense in inflating the locks 200x for one insert
	//if (ratio > 15) ratio = 15;
	//printf("Dividing ratio %d\n", ratio);

	
	downcast<<<(count-1)/512+1, 512>>>(count, vals, rsqf_inserts);


  //insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);

	//__host__ float insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);
	insertGPU(test_cqf_gpu, count, rsqf_inserts, rsqf_returns);
	
  //cudaMemset((uint64_t *) buffer_sizes, 0, ratio*num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_one_hash(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK/ratio, num_locks*ratio, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
  //bulk_insert_bucketing_buffer_provided_timed(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_no_atomics(g_quotient_filter, vals,0,1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_sizes);

	cudaDeviceSynchronize();
	return 0;
}


__global__ void rsqf_check(int * returns, uint64_t count, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= count) return;

	if (returns[tid] == UINT_MAX){
		atomicAdd((unsigned long long int *) misses, (unsigned long long int) 1);
	}

}

extern inline uint64_t rsqf_bulk_get(uint64_t * vals, uint64_t count){


	// uint64_t * misses;
 //  cudaMallocManaged((void **)& misses, sizeof(uint64_t));
 //  misses[0] = 0;

  downcast<<<(count-1)/512+1, 512>>>(count, vals, rsqf_inserts);
  //return bulk_get_wrapper(g_quotient_filter, vals, count);
  launchUnsortedLookups(test_cqf_gpu, count, rsqf_inserts, rsqf_returns);
  cudaDeviceSynchronize();


  //rsqf_check<<<(count-1)/512+1, 512>>>(rsqf_returns, count, misses);
  //cudaDeviceSynchronize();

  //uint64_t toReturn = misses[0];
  uint64_t toReturn = 0;  

  //cudaFree(misses);

  return toReturn;


}

extern inline uint64_t rsqf_bulk_get_fp(uint64_t * vals, uint64_t count){


  uint64_t * misses;
  cudaMallocManaged((void **)& misses, sizeof(uint64_t));
  misses[0] = 0;

  downcast<<<(count-1)/512+1, 512>>>(count, vals, rsqf_inserts);
  //return bulk_get_wrapper(g_quotient_filter, vals, count);
  launchUnsortedLookups(test_cqf_gpu, count, rsqf_inserts, rsqf_returns);
  cudaDeviceSynchronize();


  rsqf_check<<<(count-1)/512+1, 512>>>(rsqf_returns, count, misses);
  cudaDeviceSynchronize();

  uint64_t toReturn = misses[0];
  //uint64_t toReturn = 0;  

  cudaFree(misses);

  return toReturn;


}


extern inline void rsqf_bulk_delete(uint64_t * vals, uint64_t count){


	return;
}



#endif

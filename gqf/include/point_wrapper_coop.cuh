/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef POINT_WRAPPER_COOP_CUH
#define POINT_WRAPPER_COOP_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gqf.cuh"
#include "gqf_int.cuh"
#include "gqf_file.cuh"

#include <chrono>

QF* coop_point_quotient_filter;


extern inline int coop_point_init(uint64_t nbits, uint64_t hash, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	//consolidate all of the device construction into one convenient func!
	qf_malloc_device(&coop_point_quotient_filter, nbits, false);

	

	return 0;
}

extern inline int coop_point_destroy()
{
	//fix me this isn't going to work
	
	qf_destroy_device(coop_point_quotient_filter);
	return 0;
}



extern inline int coop_point_bulk_insert_wrapper(uint64_t * vals, uint64_t count)
{

	//calculate ratios
	//total_items += count;

	//int ratio = num_slots/total_items;

	//no sense in inflating the locks 200x for one insert
	//if (ratio > 15) ratio = 15;
	//printf("Dividing ratio %d\n", ratio);
	
  //cudaMemset((uint64_t *) buffer_sizes, 0, ratio*num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_one_hash(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK/ratio, num_locks*ratio, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);

	const int BLOCK_SIZE = 32;




    point_bulk_insert_cooperative<<<(count*32-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(coop_point_quotient_filter, vals, count);
	
	cudaDeviceSynchronize();
	return 0;
}

//dummy func
extern inline uint64_t coop_point_bulk_get(uint64_t * vals, uint64_t count){


	auto start = std::chrono::high_resolution_clock::now();


	uint64_t toReturn = point_get_wrapper(coop_point_quotient_filter, vals, count);

	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();


 	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << count << " in " << diff.count() << " seconds\n";


	return toReturn;


}

extern inline void coop_point_bulk_delete(uint64_t * vals, uint64_t count){

	return;
}


#endif

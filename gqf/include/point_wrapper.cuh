/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef POINT_WRAPPER_CUH
#define POINT_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gqf.cuh"
#include "gqf_int.cuh"
#include "gqf_file.cuh"


QF* point_quotient_filter;


extern inline int point_init(uint64_t nbits, uint64_t hash, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	//consolidate all of the device construction into one convenient func!
	qf_malloc_device(&point_quotient_filter, nbits, false);

	

	return 0;
}

extern inline int point_destroy()
{
	//fix me this isn't going to work
	
	qf_destroy_device(point_quotient_filter);
	return 0;
}



extern inline int point_bulk_insert_wrapper(uint64_t * vals, uint64_t count)
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
    point_bulk_insert<<<(count-1)/32+1,32>>>(point_quotient_filter, vals, count);
	
	cudaDeviceSynchronize();
	return 0;
}

//dummy func
extern inline uint64_t point_bulk_get(uint64_t * vals, uint64_t count){

	return point_get_wrapper(point_quotient_filter, vals, count);


}

extern inline uint64_t point_bulk_get_fp(uint64_t * vals, uint64_t count){

	return point_get_wrapper_fp(point_quotient_filter, vals, count);


}

extern inline void point_bulk_delete(uint64_t * vals, uint64_t count){

	return;
}


#endif

/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>  
 *                  Hunter McCoy <hjmccoy@lbl.gov>  
 *
 * ============================================================================
 */

#ifndef GQF_WRAPPER_CUH
#define GQF_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gqf.cuh"
#include "gqf_int.cuh"

QF* g_quotient_filter;


uint64_t num_slots;
uint64_t total_items;
	
uint64_t ** buffers;
	
uint64_t * buffer_backing;

uint64_t item_cap, current_items;

uint64_t xnslots;


extern inline int gqf_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	qf_malloc_device(& g_quotient_filter, nbits, true);


	return 0;
}


extern inline int gqf_destroy()
{
	//fix me this isn't going to work
	//free_buffers_premalloced(g_quotient_filter, buffers, buffer_backing, buffer_sizes, num_locks);
	//qf_free_gpu(g_quotient_filter);
	qf_destroy_device(g_quotient_filter);
	

	return 0;
}




extern inline int gqf_bulk_insert(uint64_t * vals, uint64_t count)
{

	bulk_insert(g_quotient_filter, count, vals, QF_NO_LOCK);

	//bulk_insert_timed(g_quotient_filter, count, vals, QF_NO_LOCK);

	cudaDeviceSynchronize();
	return 0;
}

extern inline uint64_t gqf_bulk_get(uint64_t * vals, uint64_t count){

  
	//__host__ uint64_t bulk_get_nocount_wrapper(QF * qf, uint64_t * vals, uint64_t nvals)
  //return bulk_get_misses_wrapper(g_quotient_filter, vals, count);
  return bulk_get_nocount_wrapper(g_quotient_filter, vals, count);

}


extern inline void gqf_bulk_delete(uint64_t * vals, uint64_t count){

	bulk_delete(g_quotient_filter, count, vals, QF_NO_LOCK);


	cudaDeviceSynchronize();
	return;
}

extern inline uint64_t gqf_bulk_get_fp(uint64_t * vals, uint64_t count){

  
	//__host__ uint64_t bulk_get_nocount_wrapper(QF * qf, uint64_t * vals, uint64_t nvals)
  //return bulk_get_misses_wrapper(g_quotient_filter, vals, count);

  //bulk_get_misses(QF * qf, uint64_t * vals,  uint64_t nvals, uint64_t key_count, uint64_t * counter, uint8_t flags){
  return bulk_get_misses_wrapper(g_quotient_filter, vals, count);

}

#endif

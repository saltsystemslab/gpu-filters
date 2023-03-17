/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>  
 *                  Hunter McCoy <hjmccoy@lbl.gov>  
 *
 * ============================================================================
 */

#ifndef COOP_GQF_WRAPPER_CUH
#define COOP_GQF_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gqf.cuh"
#include "gqf_int.cuh"

QF* coop_g_quotient_filter;


extern inline int coop_gqf_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	qf_malloc_device(& coop_g_quotient_filter, nbits, true);


	return 0;
}


extern inline int coop_gqf_destroy()
{
	//fix me this isn't going to work
	//free_buffers_premalloced(coop_g_quotient_filter, buffers, buffer_backing, buffer_sizes, num_locks);
	//qf_free_gpu(coop_g_quotient_filter);
	qf_destroy_device(coop_g_quotient_filter);
	

	return 0;
}




extern inline int coop_gqf_bulk_insert(uint64_t * vals, uint64_t count)
{

	bulk_insert_cooperative(coop_g_quotient_filter, count, vals, QF_NO_LOCK);

	cudaDeviceSynchronize();
	return 0;
}

extern inline uint64_t coop_gqf_bulk_get(uint64_t * vals, uint64_t count){

  

  //return bulk_get_misses_wrapper(coop_g_quotient_filter, vals, count);

  return cooperative_bulk_get_wrapper(coop_g_quotient_filter, vals, count);

}

extern inline uint64_t coop_gqf_bulk_get_fp(uint64_t * vals, uint64_t count){

  

  //return bulk_get_misses_wrapper(coop_g_quotient_filter, vals, count);

  return cooperative_bulk_get_wrapper(coop_g_quotient_filter, vals, count);

}



extern inline void coop_gqf_bulk_delete(uint64_t * vals, uint64_t count){

	bulk_delete(coop_g_quotient_filter, count, vals, QF_NO_LOCK);

	cudaDeviceSynchronize();
	return;
}


#endif

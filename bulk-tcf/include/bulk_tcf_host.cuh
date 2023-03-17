#ifndef BULK_TCF_H 
#define BULK_TCF_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "bulk_tcf_metadata.cuh"
#include "bulk_tcf_key_val_pair.cuh"
#include "templated_block.cuh"
#include "bulk_tcf_hashutil.cuh"
#include "templated_sorting_funcs.cuh"
#include <stdio.h>
#include <assert.h>

//thrust stuff
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>


#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>


namespace cg = cooperative_groups;


	
//the bulk-tcf exists on device but needs several bits of metadata that must exist on host
// in addition, all functions are called on host, as specialized group kernels must be used for the bulk-TCF
// this host wrapper encapsulates the complicated behavior and exposes a few easy-to-use functions
// If you want to run it solely as a device pointer, that is still possible using "bulk_tcf.cuh"

template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
struct host_bulk_tcf{

	using bulk_tcf_type = bulk_tcf<Key, Val, Wrapper>;
	using internal_key_type = key_val_pair<Key, Val, Wrapper>;

	bulk_tcf_type * dev_tcf;

	uint64_t num_teams;
	uint64_t num_blocks;


	static host_bulk_tcf * build_tcf(uint64_t nitems){


		host_bulk_tcf * host_version;

		cudaMallocHost((void**)&host_version, sizeof(host_bulk_tcf));


		host_version->dev_tcf = build_tcf(nitems);

		host_version->num_blocks = host_version->dev_tcf->get_num_blocks();

		host_version->num_teams = host_version->dev_tcf->get_num_teams();


		return host_version;


	}


	__host__ void bulk_insert(Large_Keys * host_large_keys, uint64_t nitems, uint64_t * misses){

		

		internal_key_type * dev_small_keys;

		//this may be very expensive... check cost.
		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);

		dev_tcf->attach_lossy_buffers(large_keys, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_insert(misses, num_teams);

		cudaFree(dev_small_keys);

		return;


	}

	__host__ void bulk_query(Large_Keys * host_query_keys)

	__host__ void bulk_query_values(Large_Keys * query_keys, Val * output_buffer, uint64_t nitems, uint64_t * misses)

	__host__ void bulk_delete(Large_Keys * delete_keys, uint64_t nitems, uint64_t * misses){


	}


	__host__ void associate_keys(){

	}





}


#endif //GPU_BLOCK_
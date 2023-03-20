#ifndef BULK_TCF_HOST
#define BULK_TCF_HOST


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "bulk_tcf_metadata.cuh"
#include "bulk_tcf_key_val_pair.cuh"
#include "templated_block.cuh"
#include "bulk_tcf_hashutil.cuh"
#include "templated_sorting_funcs.cuh"
#include "bulk_tcf.cuh"
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

#include <type_traits>

namespace cg = cooperative_groups;


	
//the bulk-tcf exists on device but needs several bits of metadata that must exist on host
// in addition, all functions are called on host, as specialized group kernels must be used for the bulk-TCF
// this host wrapper encapsulates the complicated behavior and exposes a few easy-to-use functions
// If you want to run it solely as a device pointer, that is still possible using "bulk_tcf.cuh"

template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
struct host_bulk_tcf {


	using my_type = host_bulk_tcf<Large_Keys, Key, Val, Wrapper>;
	using bulk_tcf_type = bulk_tcf<Key, Val, Wrapper>;
	using internal_key_type = key_val_pair<Key, Val, Wrapper>;

	bulk_tcf_type * dev_tcf;

	uint64_t num_teams;
	uint64_t num_blocks;


	__host__ static my_type * host_build_tcf(uint64_t nslots){


		host_bulk_tcf * host_version;

		cudaMallocHost((void**)&host_version, sizeof(host_bulk_tcf));


		host_version->dev_tcf = build_tcf<Key, Val, Wrapper>(nslots);

		host_version->num_blocks = host_version->dev_tcf->get_num_blocks();

		host_version->num_teams = host_version->dev_tcf->get_num_teams();


		return host_version;


	}

	__host__ static void host_free_tcf(my_type * host_version){


		free_tcf<Key, Val, Wrapper>(host_version->dev_tcf);


		cudaFreeHost(host_version);

	}


	__host__ void bulk_insert(Large_Keys * large_keys, uint64_t nitems, uint64_t * misses){

		

		internal_key_type * dev_small_keys;

		//this may be very expensive... check cost.
		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);

		dev_tcf->attach_lossy_buffers(large_keys, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_insert(misses, num_teams);

		cudaFree(dev_small_keys);

		return;


	}


	__host__ void bulk_insert_values(Large_Keys * large_keys, Val * vals, uint64_t nitems, uint64_t * misses){

		

		internal_key_type * dev_small_keys;

		//this may be very expensive... check cost.
		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);

		dev_tcf->attach_lossy_buffers_vals(large_keys, vals, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_insert(misses, num_teams);

		cudaFree(dev_small_keys);

		return;


	}


	__host__ bool * bulk_query(Large_Keys * host_query_keys, uint64_t nitems){
		internal_key_type * dev_small_keys;

		bool * scrambled_hits;

		cudaMalloc((void **)&scrambled_hits, sizeof(bool)*nitems);

		cudaMemset(scrambled_hits, 0, sizeof(bool)*nitems);


		uint64_t * indices;

		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);
		cudaMalloc((void **)&indices, sizeof(uint64_t)*nitems);

		dev_tcf->attach_lossy_buffers_recovery(host_query_keys, indices, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_query(scrambled_hits, num_teams);

		bool * return_hits;

		cudaMalloc((void **)&return_hits, sizeof(bool)*nitems);

		cast_hits<<<(nitems-1)/512+1,512>>>(return_hits, scrambled_hits, indices, nitems);

		cudaDeviceSynchronize();

		cudaFree(dev_small_keys);

		cudaFree(scrambled_hits);
		cudaFree(indices);

		return return_hits;

	}	

	//if you don't care about queries order than use this! It's slightly faster.
	__host__ bool * bulk_query_scrambled(Large_Keys * host_query_keys, uint64_t nitems){
		internal_key_type * dev_small_keys;

		bool * scrambled_hits;

		cudaMalloc((void **)&scrambled_hits, sizeof(bool)*nitems);

		cudaMemset(scrambled_hits, 0, sizeof(bool)*nitems);


		uint64_t * indices;

		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);
		cudaMalloc((void **)&indices, sizeof(uint64_t)*nitems);

		dev_tcf->attach_lossy_buffers_recovery(host_query_keys, indices, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_query(scrambled_hits, num_teams);

		cudaFree(dev_small_keys);

		cudaDeviceSynchronize();

		return scrambled_hits;


	}	

	//lookup keys, marking hits and storing values in the output buffer
	__host__ bool * bulk_query_values(Large_Keys * query_keys, Val * output_buffer, uint64_t nitems){

		//assert only called when running.
		//assert(!std::is_same<Val, empty>);


		internal_key_type * dev_small_keys;

		bool * scrambled_hits;

		cudaMalloc((void **)&scrambled_hits, sizeof(bool)*nitems);

		cudaMemset(scrambled_hits, 0, sizeof(bool)*nitems);


		uint64_t * indices;

		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);
		cudaMalloc((void **)&indices, sizeof(uint64_t)*nitems);


		dev_tcf->attach_lossy_buffers_recovery(query_keys, indices, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_query_values(scrambled_hits, num_teams);


		bool * return_hits;

		cudaMalloc((void **)&return_hits, sizeof(bool)*nitems);

		cast_hits_values<internal_key_type, Val><<<(nitems-1)/512+1,512>>>(return_hits, scrambled_hits, indices, dev_small_keys, output_buffer, nitems);

		cudaDeviceSynchronize();

		cudaFree(dev_small_keys);
		cudaFree(scrambled_hits);
		cudaFree(indices);

		return return_hits;


	}


	__host__ void check_correctness(Large_Keys * keys, uint64_t nitems){

		dev_tcf->check_correctness(keys, nitems);


	}

	__host__ uint64_t get_fill(){

		uint64_t * counter;

		cudaMallocManaged((void **)&counter, sizeof(uint64_t));

		cudaDeviceSynchronize();

		counter[0] = 0;

		dev_tcf->get_fill(counter, num_blocks);

		cudaDeviceSynchronize();

		uint64_t return_value = counter[0];

		cudaFree(counter);
		
		return return_value;


	}

	__host__ bool * bulk_delete_scrambled(Large_Keys * delete_keys, uint64_t nitems){


		internal_key_type * dev_small_keys;


		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);

		bool * scrambled_hits;

		cudaMalloc((void **)&scrambled_hits, sizeof(bool)*nitems);

		cudaMemset(scrambled_hits, 0, sizeof(bool)*nitems);


		dev_tcf->attach_lossy_buffers(delete_keys, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_delete(scrambled_hits, num_teams);

		return scrambled_hits;

	}


	__host__ bool * bulk_delete(Large_Keys * delete_keys, uint64_t nitems){


		internal_key_type * dev_small_keys;


		cudaMalloc((void **)&dev_small_keys, sizeof(internal_key_type)*nitems);

		uint64_t * indices;

		cudaMalloc((void **)&indices, sizeof(uint64_t)*nitems);

		bool * scrambled_hits;

		cudaMalloc((void **)&scrambled_hits, sizeof(bool)*nitems);

		cudaMemset(scrambled_hits, 0, sizeof(bool)*nitems);


		dev_tcf->attach_lossy_buffers_recovery(delete_keys, indices, dev_small_keys, nitems, num_blocks);

		dev_tcf->bulk_delete(scrambled_hits, num_teams);

		bool * return_hits;

		cudaMalloc((void **)&return_hits, sizeof(bool)*nitems);

		cast_hits<<<(nitems-1)/512+1,512>>>(return_hits, scrambled_hits, indices, nitems);

		cudaDeviceSynchronize();

		cudaFree(scrambled_hits);

		cudaFree(indices);

		return return_hits;

	}




};


#endif //GPU_BLOCK_
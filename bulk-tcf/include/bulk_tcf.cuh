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


//counters are now external to allow them to permanently reside in the l1 cache.
//this should improve performance and allow for different loading schemes
//that are less reliant on the initial load.

//these are templated on just one thing
//key_value_pairs

// template <typename Tag_type>
// __device__ bool assert_sorted(Tag_type * tags, int nitems){


// 	if (nitems < 1) return true;

// 	Tag_type smallest = tags[0];

// 	for (int i=1; i< nitems; i++){

// 		if (tags[i] < smallest) return false;

// 		smallest = tags[i];
// 	}

// 	return true;

// }


//cuda templated globals


//debugging kernels
template <typename Filter, typename Key_type>
__global__ void setup_correctness_kernel(Filter * my_tcf, uint64_t * items, uint64_t * bucket_ids, Key_type * tags, int * alt_bucket_ids, uint64_t nitems){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid>=nitems) return;


	uint64_t key = items[tid];

	key = my_tcf->hash_key(key);

	tags[tid].set_key(key);

	uint64_t bucket_index = my_tcf->get_bucket_from_reference(key);

	int remainder = my_tcf->get_alt_bucket_from_key(tags[tid], bucket_index % BLOCKS_PER_THREAD_BLOCK) % BLOCKS_PER_THREAD_BLOCK;

	if (bucket_index % BLOCKS_PER_THREAD_BLOCK == remainder){
		remainder = (remainder+1) % BLOCKS_PER_THREAD_BLOCK;
	}

	bucket_ids[tid] = bucket_index;

	alt_bucket_ids[tid] = remainder;



}


template <typename Filter, typename Key_type>
__global__ void check_correctness_kernel(Filter * my_tcf, uint64_t * bucket_ids, Key_type * tags, int * alt_bucket_ids, uint64_t nitems){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid>=nitems) return;

	uint64_t my_bucket_index = bucket_ids[tid];

	Key_type my_tag = tags[tid];

	int my_alt_bucket = alt_bucket_ids[tid];


	for (uint64_t i = 0; i< nitems; i++){


		if (bucket_ids[i] == my_bucket_index){

			if (tags[i] == my_tag){

				if (alt_bucket_ids[i] != my_alt_bucket){

					printf("Check failed for indices %llu, %llu\n", tid, i);
				}

			}

		}



	}


}


//sorts the set
template <typename Filter, typename Key_type>
__global__ void hash_all_key_purge(Filter * my_tcf, uint64_t * keys, Key_type * tags, uint64_t nvals){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;

	uint64_t key = keys[tid];

	//shrink the keys
	//this is valid and preserves query values.

	key = my_tcf->hash_key(key);

	tags[tid].set_key(key);

	tags[tid].mark_primary();


	//uint64_t hashed_key = my_tcf->hash_key(keys[tid]);

	//key % num_blocks*(1ULL << tag_bits);

	uint64_t bucket_index = my_tcf->get_bucket_from_reference(key);
	
						//bucket << 16
	uint64_t new_key = my_tcf->get_reference_from_bucket(bucket_index) | tags[tid].get_key();

	//buckets are now sortable!
	keys[tid] = new_key;


}


__global__ void cast_hits(bool * hits, bool * scrambled_hits, uint64_t * indices, uint64_t nitems){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >= nitems) return;

	hits[indices[tid]] = scrambled_hits[tid];

	return;

}

template <typename Filter, typename Key_type>
__global__ void hash_all_key_purge_index(Filter * my_tcf, uint64_t * large_keys, uint64_t * indices, Key_type * tags, uint64_t nvals){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;

	uint64_t key = large_keys[tid];


	key = my_tcf->hash_key(key);

	//shrink the keys
	//this is valid and preserves query values.
	tags[tid].set_key(key);

	tags[tid].mark_primary();


	//uint64_t hashed_key = my_tcf->hash_key(large_keys[tid]);

	uint64_t bucket_index = my_tcf->get_bucket_from_reference(key);

	uint64_t new_key = my_tcf->get_reference_from_bucket(bucket_index) | tags[tid].get_key();

	//quick assertion check

	// Key_type old_key = keys[tid];

	// keys[tid].set_key(new_key);

	// if (old_key != keys[tid]){
	// 	printf("Conversion err\n");
	// }

	//buckets are now sortable!
	large_keys[tid] = new_key;

	indices[tid] = tid;


}
	
//overwrite the old keys with new ones from the current list.
template <typename Filter, typename Key_type>
__global__ void associate_keys_with_indices(Filter * my_tcf, uint64_t * large_keys, Key_type * keys, uint64_t nvals){

	uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

	if (tid >=nvals) return;

	//is this valid
	keys[tid].set_key(large_keys[tid]);

}

template <typename Filter, typename Key_type>
__global__ void new_hash_all_key_purge(Filter * my_tcf, uint64_t * large_keys, Key_type * keys, uint64_t nvals){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;

	uint64_t key = large_keys[tid];

	//shrink the keys
	//this is valid and preserves query values.

	key = my_tcf->hash_key(key);

	//need to get leftovers..
	keys[tid].set_key(my_tcf->get_remainder_from_hash(key));


	//uint64_t hashed_key = my_tcf->hash_key(large_keys[tid]);

	//key = my_tcf->hash_key(key);

	//uint64_t new_key = my_tcf->get_reference_from_bucket(key) | keys[tid].get_key();

	//buckets are now sortable!
	large_keys[tid] = key;


}


template <typename Filter, typename Key_type>
__global__ void hash_all_key_purge_static(uint64_t * large_keys, Key_type * keys, uint64_t nvals, uint64_t ext_num_blocks){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;

	uint64_t key = large_keys[tid];

	//shrink the keys
	keys[tid] = (Key_type) large_keys[tid];


	key = Filter::static_hash_key(key, ext_num_blocks);

	uint64_t new_key = Filter::static_get_reference_from_bucket(key) | keys[tid].get_key();

	//buckets are now sortable!
	large_keys[tid] = new_key;

}





// template<typename Filter, typename Key_type>
// __global__ void hash_all_keys(Filter * my_tcf, uint64_t * keys, uint64_t nvals){

// 	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

// 	if (tid >= nvals) return;

// 	uint64_t key = keys[tid];

// 	uint64_t hash = my_tcf->hash_key(key);

// 	uint64_t new_key = my_tcf->get_reference_from_bucket(hash) | key;

// 	keys[tid] = new_key;
// }


template <typename Filter, typename Key_type>
__global__ void hash_all_key_purge_cycles(Filter * my_tcf, uint64_t * vals, Key_type * keys, uint64_t nvals, uint64_t * cycles){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nvals) return;

	uint64_t clock_start = clock();

	uint64_t key = vals[tid];

	keys[tid] = (Key_type) vals[tid];

	key = my_tcf->hash_key(key);

	uint64_t new_key = my_tcf->get_reference_from_bucket(key) | keys[tid].get_key();

	//buckets are now sortable!
	vals[tid] = new_key;

	uint64_t clock_end = clock();

	if (threadIdx.x % 32 == 0){

		atomicAdd((unsigned long long int *)&cycles[3], clock_end-clock_start);

	}

}


template <typename Filter, typename Key_type>
__global__ void set_buffers_binary(Filter * my_tcf, uint64_t * references, Key_type * keys, uint64_t nvals){


		// #if DEBUG_ASSERTS

		// assert(assert_sorted(keys, nvals));

		// #endif


		uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= my_tcf->num_blocks) return;

		//uint64_t slots_per_lock = VIRTUAL_BUCKETS;

		//since we are finding all boundaries, we only need

		//printf("idx %llu\n", idx);

		//this sounds right? - they divide to go back so I think this is fine
		//this is fine but need to apply a hash
		uint64_t boundary = idx; //<< qf->metadata->bits_per_slot;


		//This is the code I'm stealing that assumption from
		//uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);	
		//uint64_t lock_index = hash_bucket_index / slots_per_lock;


		uint64_t lower = 0;
		uint64_t upper = nvals ;
		uint64_t index = upper-lower;

		//upper is non inclusive bound


		//if we exceed bounds that's our index
		while (upper != lower){


			index = lower + (upper - lower)/2;

			//((keys[index] >> TAG_BITS)
			uint64_t bucket = my_tcf->get_bucket_from_reference(references[index]);


			if (index != 0)
			uint64_t old_bucket = my_tcf->get_bucket_from_reference(references[index-1]);

			if (bucket < boundary){

				//false - the list before this point can be removed
				lower = index+1;

				//jump to a new midpoint
				


			} else if (index==0){

				//will this fix? otherwise need to patch via round up
				upper = index;

				//(get_bucket_from_reference(references[index-1])
				//(keys[index-1] >> TAG_BITS)

			} else if (my_tcf->get_bucket_from_reference(references[index-1]) < boundary) {

				//set index! this is the first instance where I am valid and the next isnt
				//buffers[idx] = keys+index;
				break;

			} else {

				//we are too far right, all keys to the right do not matter
				upper = index;


			}

		}

		//we either exited or have an edge condition:
		//upper == lower iff 0 or max key
		index = lower + (upper - lower)/2;

		//assert(my_tcf->get_bucket_from_hash(keys[index]) <= idx);


		my_tcf->buffers[idx] = keys + index;
		


}

//new version
//uses a different space partitioning scheme.
//hashes should be evenly distributed across ~0ULL space
template <typename Filter, typename Key_type>
__global__ void new_set_buffers_binary(Filter * my_tcf, uint64_t * references, Key_type * keys, uint64_t nvals){


		// #if DEBUG_ASSERTS

		// assert(assert_sorted(keys, nvals));

		// #endif


		uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= my_tcf->num_blocks) return;

		//uint64_t slots_per_lock = VIRTUAL_BUCKETS;

		//since we are finding all boundaries, we only need

		//printf("idx %llu\n", idx);

		//this sounds right? - they divide to go back so I think this is fine
		//this is fine but need to apply a hash
		uint64_t boundary = idx; //<< qf->metadata->bits_per_slot;


		//This is the code I'm stealing that assumption from
		//uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);	
		//uint64_t lock_index = hash_bucket_index / slots_per_lock;


		uint64_t lower = 0;
		uint64_t upper = nvals ;
		uint64_t index = upper-lower;

		//upper is non inclusive bound


		//if we exceed bounds that's our index
		while (upper != lower){


			index = lower + (upper - lower)/2;

			//((keys[index] >> TAG_BITS)
			uint64_t bucket = my_tcf->get_bucket_from_hash(references[index]);


			if (index != 0)
			uint64_t old_bucket = my_tcf->get_bucket_from_hash(references[index-1]);

			if (bucket < boundary){

				//false - the list before this point can be removed
				lower = index+1;

				//jump to a new midpoint
				


			} else if (index==0){

				//will this fix? otherwise need to patch via round up
				upper = index;

				//(get_bucket_from_reference(references[index-1])
				//(keys[index-1] >> TAG_BITS)

			} else if (my_tcf->get_bucket_from_hash(references[index-1]) < boundary) {

				//set index! this is the first instance where I am valid and the next isnt
				//buffers[idx] = keys+index;
				break;

			} else {

				//we are too far right, all keys to the right do not matter
				upper = index;


			}

		}

		//we either exited or have an edge condition:
		//upper == lower iff 0 or max key
		index = lower + (upper - lower)/2;

		//assert(my_tcf->get_bucket_from_hash(keys[index]) <= idx);


		my_tcf->buffers[idx] = keys + index;
		


}


template <typename Filter, typename Key_type>
__global__ void set_buffers_binary(Filter * my_tcf, uint64_t * keys, uint64_t nvals){


		// #if DEBUG_ASSERTS

		// assert(assert_sorted(keys, nvals));

		// #endif


		uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= my_tcf->num_blocks) return;

		//uint64_t slots_per_lock = VIRTUAL_BUCKETS;

		//since we are finding all boundaries, we only need

		//printf("idx %llu\n", idx);

		//this sounds right? - they divide to go back so I think this is fine
		//this is fine but need to apply a hash
		uint64_t boundary = idx; //<< qf->metadata->bits_per_slot;


		//This is the code I'm stealing that assumption from
		//uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);	
		//uint64_t lock_index = hash_bucket_index / slots_per_lock;


		uint64_t lower = 0;
		uint64_t upper = nvals ;
		uint64_t index = upper-lower;

		//upper is non inclusive bound


		//if we exceed bounds that's our index
		while (upper != lower){


			index = lower + (upper - lower)/2;

			//((keys[index] >> TAG_BITS)
			uint64_t bucket = my_tcf->get_bucket_from_reference(keys[index]);


			if (index != 0)
			uint64_t old_bucket = my_tcf->get_bucket_from_reference(keys[index-1]);

			if (bucket < boundary){

				//false - the list before this point can be removed
				lower = index+1;

				//jump to a new midpoint
				


			} else if (index==0){

				//will this fix? otherwise need to patch via round up
				upper = index;

				//(get_bucket_from_reference(references[index-1])
				//(keys[index-1] >> TAG_BITS)

			} else if (my_tcf->get_bucket_from_reference(keys[index-1]) < boundary) {

				//set index! this is the first instance where I am valid and the next isnt
				//buffers[idx] = keys+index;
				break;

			} else {

				//we are too far right, all keys to the right do not matter
				upper = index;


			}

		}

		//we either exited or have an edge condition:
		//upper == lower iff 0 or max key
		index = lower + (upper - lower)/2;

		//assert(my_tcf->get_bucket_from_hash(keys[index]) <= idx);


		my_tcf->buffers[idx] = keys + index;
		


}

template <typename Filter, typename Key_type>
__global__ void set_buffers_binary_cycles(Filter * my_tcf, uint64_t * references, Key_type * keys, uint64_t nvals, uint64_t * cycles){


		// #if DEBUG_ASSERTS

		// assert(assert_sorted(keys, nvals));

		// #endif


		uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= my_tcf->num_blocks) return;

		uint64_t clock_start = clock();

		//uint64_t slots_per_lock = VIRTUAL_BUCKETS;

		//since we are finding all boundaries, we only need

		//printf("idx %llu\n", idx);

		//this sounds right? - they divide to go back so I think this is fine
		//this is fine but need to apply a hash
		uint64_t boundary = idx; //<< qf->metadata->bits_per_slot;


		//This is the code I'm stealing that assumption from
		//uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);	
		//uint64_t lock_index = hash_bucket_index / slots_per_lock;


		uint64_t lower = 0;
		uint64_t upper = nvals ;
		uint64_t index = upper-lower;

		//upper is non inclusive bound


		//if we exceed bounds that's our index
		while (upper != lower){


			index = lower + (upper - lower)/2;

			//((keys[index] >> TAG_BITS)
			uint64_t bucket = my_tcf->get_bucket_from_reference(references[index]);


			if (index != 0)
			uint64_t old_bucket = my_tcf->get_bucket_from_reference(references[index-1]);

			if (bucket < boundary){

				//false - the list before this point can be removed
				lower = index+1;

				//jump to a new midpoint
				


			} else if (index==0){

				//will this fix? otherwise need to patch via round up
				upper = index;

				//(get_bucket_from_reference(references[index-1])
				//(keys[index-1] >> TAG_BITS)

			} else if (my_tcf->get_bucket_from_reference(references[index-1]) < boundary) {

				//set index! this is the first instance where I am valid and the next isnt
				//buffers[idx] = keys+index;
				break;

			} else {

				//we are too far right, all keys to the right do not matter
				upper = index;


			}

		}

		//we either exited or have an edge condition:
		//upper == lower iff 0 or max key
		index = lower + (upper - lower)/2;

		//assert(my_tcf->get_bucket_from_hash(keys[index]) <= idx);


		my_tcf->buffers[idx] = keys + index;

		uint64_t clock_end = clock();

		if (threadIdx.x % 32 == 0){

		atomicAdd((unsigned long long int *)&cycles[4], clock_end-clock_start);

		}


		


}

template <typename Filter, typename Key_type>
__global__ void set_buffer_lens(Filter * my_tcf, uint64_t num_keys,  Key_type * keys){


	// #if DEBUG_ASSERTS

	// assert(assert_sorted(keys, num_keys));

	// #endif

	uint64_t num_buffers = my_tcf->num_blocks;


	uint64_t idx = threadIdx.x + blockDim.x*blockIdx.x;

	if (idx >= num_buffers) return;


	//only 1 thread will diverge - should be fine - any cost already exists because of tail
	if (idx != num_buffers-1){

		//this should work? not 100% convinced but it seems ok
		my_tcf->buffer_sizes[idx] = my_tcf->buffers[idx+1] - my_tcf->buffers[idx];
	} else {

		my_tcf->buffer_sizes[idx] = num_keys - (my_tcf->buffers[idx] - keys);

	}

	return;


}

template <typename Filter, typename Key_type>
__global__ void set_buffer_lens_cycles(Filter * my_tcf, uint64_t num_keys,  Key_type * keys, uint64_t * cycles){


	// #if DEBUG_ASSERTS

	// assert(assert_sorted(keys, num_keys));

	// #endif

	uint64_t num_buffers = my_tcf->num_blocks;


	uint64_t idx = threadIdx.x + blockDim.x*blockIdx.x;

	if (idx >= num_buffers) return;

	uint64_t clock_start = clock();


	//only 1 thread will diverge - should be fine - any cost already exists because of tail
	if (idx != num_buffers-1){

		//this should work? not 100% convinced but it seems ok
		my_tcf->buffer_sizes[idx] = my_tcf->buffers[idx+1] - my_tcf->buffers[idx];
	} else {

		my_tcf->buffer_sizes[idx] = num_keys - (my_tcf->buffers[idx] - keys);

	}

	uint64_t clock_end = clock();

	if (threadIdx.x % 32 == 0){

		atomicAdd((unsigned long long int *)&cycles[5], clock_end-clock_start);

	}

	return;


}

template <typename Filter>
__global__ void sorted_bulk_insert_kernel(Filter * tcf, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	uint64_t teamID = tid / (BLOCK_SIZE);



	//TODO double check me
	if (teamID >= tcf->num_teams) return;


	//tcf->sorted_mini_filter_block(misses);

	tcf->sorted_dev_insert(misses);
	//tcf->persistent_dev_insert(misses);

	return;

	


}


template <typename Filter>
__global__ void bulk_get_fill_kernel(Filter * tcf, uint64_t * counter, uint64_t num_blocks){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid >=num_blocks) return;


	//TODO double check m


	//tcf->sorted_mini_filter_block(misses);

	uint64_t fill = tcf->get_block_fill(tid);

	atomicAdd((unsigned long long int *)counter, fill);
	//tcf->persistent_dev_insert(misses);

	return;

	


}

template <typename Filter>
__global__ void sorted_bulk_insert_kernel_cycles(Filter * tcf, uint64_t * misses, uint64_t * cycles){

	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;


	uint64_t teamID = tid / (BLOCK_SIZE);



	//TODO double check me
	if (teamID >= tcf->num_teams) return;


	//tcf->sorted_mini_filter_block(misses);

	tcf->sorted_dev_insert_cycles(misses, cycles);
	//tcf->persistent_dev_insert(misses);

	return;

	


}

template<typename Filter>
__global__ void bulk_sorted_query_kernel(Filter * tcf, bool * hits){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t teamID = tid / (BLOCK_SIZE);

	#if DEBUG_ASSERTS

	assert(teamID == blockIdx.x);

	#endif

	if (teamID >= tcf->num_teams) return;

	tcf->mini_filter_bulk_queries(hits);
}


template<typename Filter>
__global__ void bulk_sorted_delete_kernel(Filter * tcf, bool * hits){


	uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

	uint64_t teamID = tid / (BLOCK_SIZE);

	#if DEBUG_ASSERTS

	assert(teamID == blockIdx.x);

	#endif

	if (teamID >= tcf->num_teams) return;

	tcf->mini_filter_bulk_deletes(hits);
}


//END OF KERNELS




// template <typename Filter, typename Key_type>
// __global__ void test_kernel(T * my_tcf){


// }


template <typename T>
struct __attribute__ ((__packed__)) thread_team_block {


	T internal_blocks[BLOCKS_PER_THREAD_BLOCK];

};



template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
struct __attribute__ ((__packed__)) bulk_tcf {


	//tag bits change based on the #of bytes allocated per block

	using key_type = key_val_pair<Key, Val, Wrapper>;

	//typedef key_val_pair<Key, Val, Wrapper> key_type;

	using block_type = templated_block<key_type>;

	uint64_t num_teams;

	uint64_t num_blocks;

	uint64_t dividing_line;

	int * block_counters;

	key_type ** buffers;

	int * buffer_sizes;

	//key_type * small_keys;



	thread_team_block<block_type> * blocks;

	//static version of the code
	//for large filters,
	//we need to presort the data
	//generate presorted list and pass back.
	__host__ static void host_prep_lossy_buffers(uint64_t * host_large_keys, key_type * host_compressed_keys, uint64_t nitems, uint64_t ext_num_blocks){


		uint64_t * large_keys;
		key_type * compressed_keys;

		cudaMallocManaged((void **)&large_keys, sizeof(uint64_t)*nitems);
		cudaMallocManaged((void **)&compressed_keys, sizeof(key_type)*nitems);

		cudaMemcpy(large_keys, host_large_keys, sizeof(uint64_t)*nitems, cudaMemcpyHostToDevice);
		cudaMemcpy(compressed_keys, host_compressed_keys, sizeof(key_type)*nitems, cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		hash_all_key_purge_static<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems -1)/1024 + 1, 1024>>>(large_keys, compressed_keys, nitems, ext_num_blocks);

		thrust::sort_by_key(thrust::device, large_keys, large_keys+nitems, compressed_keys);

		cudaDeviceSynchronize();

		cudaMemcpy(host_large_keys, large_keys, sizeof(uint64_t)*nitems, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_compressed_keys, compressed_keys, sizeof(key_type)*nitems, cudaMemcpyDeviceToHost);

		cudaFree(large_keys);
		cudaFree(compressed_keys);

		return;

		return;

	}


	//static version of the code
	//for large filters,
	//we need to presort the data
	//generate presorted list and pass back.
	//this returns the time taken on device for these ops
	//you could clean it up and run the whole thing on device to get this timing
	//but I opted to measure here and convert back to host buffers so that the rest of the test file was unchanged.
	__host__ static std::chrono::duration<double> prep_lossy_buffers(uint64_t * host_large_keys, key_type * host_compressed_keys, uint64_t nitems, uint64_t ext_num_blocks){


		uint64_t * large_keys;
		key_type * compressed_keys;

		cudaMalloc((void **)&large_keys, sizeof(uint64_t)*nitems);
		cudaMalloc((void **)&compressed_keys, sizeof(key_type)*nitems);

		cudaMemcpy(large_keys, host_large_keys, sizeof(uint64_t)*nitems, cudaMemcpyHostToDevice);
		cudaMemcpy(compressed_keys, host_compressed_keys, sizeof(key_type)*nitems, cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		auto prep_start = std::chrono::high_resolution_clock::now();

		hash_all_key_purge_static<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems -1)/1024 + 1, 1024>>>(large_keys, compressed_keys, nitems, ext_num_blocks);

		thrust::sort_by_key(thrust::device, large_keys, large_keys+nitems, compressed_keys);

		cudaDeviceSynchronize();

		auto prep_end = std::chrono::high_resolution_clock::now();

		cudaMemcpy(host_large_keys, large_keys, sizeof(uint64_t)*nitems, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_compressed_keys, compressed_keys, sizeof(key_type)*nitems, cudaMemcpyDeviceToHost);

		cudaFree(large_keys);
		cudaFree(compressed_keys);

		return prep_end-prep_start;

	}


	__host__ void attach_lossy_buffers(uint64_t * large_keys, key_type * compressed_keys, uint64_t nitems, uint64_t ext_num_blocks){

		hash_all_key_purge<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems -1)/1024 + 1, 1024>>>(this, large_keys, compressed_keys, nitems);

		thrust::sort_by_key(thrust::device, large_keys, large_keys+nitems, compressed_keys);


	

		set_buffers_binary<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, large_keys, compressed_keys, nitems);

		set_buffer_lens<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, nitems, compressed_keys);


	}


	__host__ void attach_lossy_buffers_recovery(uint64_t * large_keys, uint64_t * offsets, key_type * compressed_keys, uint64_t nitems, uint64_t ext_num_blocks){

		hash_all_key_purge_index<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems -1)/1024 + 1, 1024>>>(this, large_keys, offsets, compressed_keys, nitems);

		thrust::sort_by_key(thrust::device, large_keys, large_keys+nitems, offsets);

		associate_keys_with_indices<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems -1)/1024 + 1, 1024>>>(this, large_keys, compressed_keys, nitems);


	

		set_buffers_binary<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, large_keys, compressed_keys, nitems);

		set_buffer_lens<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, nitems, compressed_keys);


	}


	__host__ void attach_presorted_buffers(uint64_t * large_keys, key_type * compressed_keys, uint64_t nitems, uint64_t ext_num_blocks){


		set_buffers_binary<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, large_keys, compressed_keys, nitems);

		set_buffer_lens<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, nitems, compressed_keys);


	}


	__host__ void attach_lossy_buffers_cycles(uint64_t * items, key_type * keys, uint64_t nitems, uint64_t ext_num_blocks, uint64_t * cycles, uint64_t * num_warps){

		hash_all_key_purge_cycles<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems -1)/1024 + 1, 1024>>>(this, items, keys, nitems, cycles);

		thrust::sort_by_key(thrust::device, items, items+nitems, keys);


	

		set_buffers_binary_cycles<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, items, keys, nitems, cycles);

		set_buffer_lens_cycles<bulk_tcf<Key, Val, Wrapper>, key_type><<<(ext_num_blocks -1)/1024+1, 1024>>>(this, nitems, keys, cycles);

		num_warps[1] += nitems/32;
		num_warps[2] += ext_num_blocks/32;
		num_warps[3] += ext_num_blocks/32;

	}

	//use uint64_t and clip them down, get better variance across the structure
	__host__ void bulk_insert_lossy_keys(uint64_t * items, key_type * keys, uint64_t nitems, uint64_t ext_num_blocks){


		attach_lossy_buffers(items, keys, nitems, ext_num_blocks);


		//sorted_mini_filter_insert(uint64_t * misses); 


	}


	__host__ void bulk_insert(uint64_t * misses, uint64_t ext_num_teams){


				sorted_bulk_insert_kernel<bulk_tcf<Key, Val, Wrapper>><<<ext_num_teams, BLOCK_SIZE>>>(this, misses);


	}

	__host__ void bulk_insert_cycles(uint64_t * misses, uint64_t * cycles, uint64_t ext_num_teams, uint64_t * num_warps){


				sorted_bulk_insert_kernel_cycles<bulk_tcf<Key, Val, Wrapper>><<<ext_num_teams, BLOCK_SIZE>>>(this, misses, cycles);

				num_warps[0] += ext_num_teams*BLOCK_SIZE/32;
	}


	//generate hash
	//upper bits of hash used for determining bucket.
	//s.t. all buckets are hashed appropriately / evenly.
	static __device__ uint64_t hash_key(uint64_t key){

		//key = MurmurHash64A(((void *)&key), sizeof(key), 42)

		//const uint key_size = (sizeof(key_type)*8);

		key = hash_64(key, ~0ULL);

		return key;
	}


	//this hash is destructive, despite only being used in one place.
	//device functions
	// __device__ uint64_t hash_key(uint64_t key){

	// 	//key = MurmurHash64A(((void *)&key), sizeof(key), 42) % (num_blocks);

	// 	key = hash_64(key, ~0ULL) % (num_blocks);

	// 	return key;

	// }

	__device__ uint64_t get_bucket_from_hash(uint64_t hash){

		//printf("Num blocks is %llu\n", num_blocks);

		const uint64_t dividing_region = (~0ULL)/num_blocks;

		return hash/dividing_region;

	}

	__device__ uint64_t get_remainder_from_hash(uint64_t hash){

		const uint64_t dividing_region = (~0ULL)/num_blocks;

		uint64_t leftover = hash % dividing_region;

		//these are all values not used by the main system. now these need to be split into
		//key range


		const uint64_t key_size = 8ULL*sizeof(Key);

		//need to compress into this many regions
		const uint64_t key_size_regions = (1ULL << key_size);

		const uint64_t big_dividing_region = (~0ULL)/(num_blocks*key_size_regions);

		if (threadIdx.x+blockIdx.x*blockDim.x == 0){
			printf("dividing_region %llu, big region %llu\n", dividing_region, big_dividing_region);
		}


		// uint64_t clipped_leftover = (~0ULL ) % dividing_region;

		// if (clipped_leftover > key_size_regions){

		// 	return hash % key_size_regions;

		// } 


		//16 bit keys
		//4 bit remainders
		//16 buckets

		//dividing region compresses into 16 options

		// so it is 0001-0000-0000-0000

		//after that modulo dividing region gives you 0000-0000-0000
		//from there, we only want the upper bits


		//if clipped is 32 bits,
		//only need 16

		//if total space is too small, use some bits from above.
		//uint64_t internal_dividing_region = (~0ULL % dividing_region)/key_size_regions;

		//printf("Leftover is %llu, max leftover is %llu\n", leftover, clipped_leftover);


		//Internal dividing region is %llu\n", internal_dividing_region);

		// uint64_t return_val = (hash % dividing_region) >> ();

		// if (return_val >= key_size_regions) printf("Bug: %llu > %llu, %f ratio\n", return_val, key_size_regions, 1.0*return_val/key_size_regions);

		// return hash/dividing_region; // % dividing_region;



	}

	__device__ uint64_t get_block_fill(uint64_t blockID){
		return block_counters[blockID];
	}

	//does this fuck it up?
	__device__ static uint64_t static_hash_key(uint64_t key, uint64_t ext_num_blocks){

		//key = MurmurHash64A(((void *)&key), sizeof(key), 42) % (ext_num_blocks);

		key = hash_64(key, ~0ULL) % (ext_num_blocks);

		return key;
	}

	//updated to use XOR trick
	__device__ uint64_t get_alt_bucket_from_key(key_type key, uint64_t bucket){

		//uint64_t new_hash = MurmurHash64A((void *)&key.get_key(), sizeof(Key), 999);

		//uint64_t new_bucket =  MurmurHash64A(((void *)&bucket), sizeof(bucket), 444);

		//return new_hash & new_bucket; 

		return (bucket ^ (key.get_key()* 0x5bd1e995));
	}

		//device functions
	__device__ uint64_t get_bucket_from_reference(uint64_t key){


		const uint key_size = 8ULL * sizeof(Key);

		if constexpr (key_size >= 64) return key % num_blocks;
		

		return (key >> (8ULL *sizeof(Key))) % num_blocks;

	}

	__device__ static uint64_t static_get_bucket_from_reference(uint64_t key, uint64_t ext_num_blocks){


		const uint key_size = 8ULL * sizeof(Key);

		if constexpr (key_size >= 64) return key % ext_num_blocks;
		
		return (key >> (8ULL *sizeof(Key)) ) % ext_num_blocks;

	}

	__device__ uint64_t get_reference_from_bucket(uint64_t hash){


		const uint key_size = 8ULL * sizeof(Key);

		if constexpr (key_size >= 64) return 0;
		
		return hash << (8ULL *sizeof(Key));

	}

	__device__ uint64_t static static_get_reference_from_bucket(uint64_t hash){


		const uint key_size = 8ULL * sizeof(Key);

		if constexpr (key_size >= 64) return 0;
		
		return hash << (8ULL *sizeof(Key));

	}

	__device__ void load_local_blocks(thread_team_block<block_type> * primary_block_ptr, int * local_counters, uint64_t blockID, int warpID, int threadID){

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			primary_block_ptr->internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];

			local_counters[i] = block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i];


		}
	}


	__device__ void unload_local_blocks(thread_team_block<block_type> * primary_block_ptr, int * local_counters, uint64_t blockID, int warpID, int threadID){

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			blocks[blockIdx.x].internal_blocks[i] = primary_block_ptr->internal_blocks[i];

			block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] = local_counters[i];

		}
	}

	//this version loads and unloads the blocks and performs all ops in shared mem
	__device__ void persistent_dev_insert(uint64_t * misses){



		cg::thread_block tb = cg::this_thread_block();

		__shared__ thread_team_block<block_type> primary_block;

		__shared__ thread_team_block<block_type> alt_storage_block;

		__shared__ int local_counters[BLOCKS_PER_THREAD_BLOCK];  

		__shared__ int buffer_offsets[BLOCKS_PER_THREAD_BLOCK];

		__shared__ int secondary_buffer_counters[BLOCKS_PER_THREAD_BLOCK];


		thread_team_block<block_type> * primary_block_ptr = &primary_block;

		thread_team_block<block_type> * alt_storage_block_ptr = &alt_storage_block;

		//load blocks

		uint64_t blockID = blockIdx.x;

		int warpID = threadIdx.x / 32;

		int threadID = threadIdx.x % 32;

		// for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		// 	primary_block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];

		// 	local_counters[i] = block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i];

		// }

		//load_local_blocks(primary_block_ptr, &local_counters[0], blockID, warpID, threadID);


		//load section
		cg::memcpy_async(tb, &primary_block, 1, blocks+blockIdx, 1);

		cg::memcpy_async(tb, local_counters, BLOCKS_PER_THREAD_BLOCK, block_counters + blockID*BLOCKS_PER_THREAD_BLOCK, BLOCKS_PER_THREAD_BLOCK);

		cg::wait(tb);


		//get counters for new_items

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		buffer_get_primary_count(primary_block_ptr, (int *)& buffer_offsets, blockID, i ,warpID, threadID);


		}
	
		dump_all_buffers_into_local_block(primary_block_ptr, alt_storage_block_ptr, &local_counters[0], &buffer_offsets[0], &secondary_buffer_counters[0], blockID, warpID, threadID, misses);


		
		thread_team_block<block_type> * temp_ptr = primary_block_ptr;
		primary_block_ptr = alt_storage_block_ptr;
		alt_storage_block_ptr = temp_ptr;


		unload_local_blocks(primary_block_ptr, &local_counters[0], blockID, warpID, threadID);

		//unload from primary ptr
		// for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

		// 	blocks[blockIdx.x].internal_blocks[i] = primary_block_ptr->internal_blocks[i];

		// 	block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] = local_counters[i];
		// }


	}


	__device__ void sorted_dev_insert(uint64_t * misses){

		__shared__ thread_team_block<block_type> block;

			//counters required
		//global offset
		//#elements dumped in round 1
		//fill within block
		//length from fill
		__shared__ int offsets[BLOCKS_PER_THREAD_BLOCK];


		__shared__ int counters[BLOCKS_PER_THREAD_BLOCK];


		//uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

		uint64_t blockID = blockIdx.x;

		int warpID = threadIdx.x / 32;

		int threadID = threadIdx.x % 32;

		//each warp should grab one block
		//TODO modify for #filter blocks per thread_team_block

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];

			//prime the offsets
			buffer_get_primary_count(&block, (int *) &offsets[0], blockID, i, threadID);



		}


		__syncthreads();


		dump_all_buffers_sorted(&block, &offsets[0], &counters[0], blockID, warpID, threadID, misses);


   		__syncthreads();

	}

	__device__ void sorted_dev_insert_cycles(uint64_t * misses, uint64_t * cycles){

		__shared__ thread_team_block<block_type> block;

			//counters required
		//global offset
		//#elements dumped in round 1
		//fill within block
		//length from fill
		__shared__ int offsets[BLOCKS_PER_THREAD_BLOCK];


		__shared__ int counters[BLOCKS_PER_THREAD_BLOCK];

		__syncthreads();

		uint64_t clock_start = clock();

		//uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

		uint64_t blockID = blockIdx.x;

		int warpID = threadIdx.x / 32;

		int threadID = threadIdx.x % 32;

		//each warp should grab one block
		//TODO modify for #filter blocks per thread_team_block

		uint64_t load_block_start = clock();

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			

			block.internal_blocks[i] = blocks[blockIdx.x].internal_blocks[i];


			



			buffer_get_primary_count(&block, (int *) &offsets[0], blockID, i, threadID);


		}


		__syncthreads();

		uint64_t load_block_end = clock();


   		if (threadID == 0){
			atomicAdd((unsigned long long int *)&cycles[6], load_block_end-load_block_start);
		}



		dump_all_buffers_sorted_cycles(&block, &offsets[0], &counters[0], blockID, warpID, threadID, misses, cycles);


   		__syncthreads();

   		uint64_t clock_end = clock();

   		if (threadID == 0){
   			atomicAdd((unsigned long long int *)&cycles[0], clock_end-clock_start);
   		}

   		


	}

	__host__ __device__ static uint64_t static_get_num_blocks(uint64_t nitems){

	uint64_t ext_num_teams = (nitems - 1)/(BLOCKS_PER_THREAD_BLOCK*block_type::max_size()) + 1;

	uint64_t ext_num_blocks = ext_num_teams*BLOCKS_PER_THREAD_BLOCK;

	return ext_num_blocks;

	}



	__device__ void buffer_get_primary_count(thread_team_block<block_type> * local_blocks, int * counters, uint64_t blockID, int warpID, int threadID){


		#if DEBUG_ASSERTS

		assert(blockID < num_teams);

		assert((blockID * BLOCKS_PER_THREAD_BLOCK + warpID) < num_blocks);
		#endif




		uint64_t global_buffer = blockID * BLOCKS_PER_THREAD_BLOCK + warpID;

		//assert(assert_sorted(buffers[global_buffer],buffer_sizes[global_buffer]));

		#if DEBUG_ASSERTS

		assert(assert_sorted(buffers[global_buffer],buffer_sizes[global_buffer]));

		assert(assert_sorted(blocks[blockID].internal_blocks[warpID].tags, block_counters[global_buffer]));

		assert(blocks_equal<key_type>(blocks[blockID].internal_blocks[warpID], local_blocks->internal_blocks[warpID], block_counters[global_buffer]));

		#endif


		int count = block_type::fill_cutoff() - block_counters[global_buffer];


		int buf_size = buffer_sizes[global_buffer];

		if (buf_size < count) count = buf_size;

		if (count < 0) count = 0;

		#if DEBUG_ASSERTS

		assert(count < block_type::max_size());

		#endif

		counters[warpID] = count;

	}


	__device__ void dump_all_buffers_sorted(thread_team_block<block_type> * local_blocks, int * offsets, int * counters, uint64_t blockID, int warpID, int threadID, uint64_t * misses){



		if (threadID == 0){

			for (int i =warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


				//remaining counters now takes into account the main list as well as new inserts

				counters[i] = offsets[i] + block_counters[blockID*BLOCKS_PER_THREAD_BLOCK + i];

				//local_block_offset;

				//start_counters[i] = 0;

			}

		}

		__syncthreads();


		#if DEBUG_ASSERTS

		for (int i = 0; i < BLOCKS_PER_THREAD_BLOCK; i++){

			assert(counters[i] <= block_type::max_size());
		}

		__syncthreads();

		#endif


		int slot;


		//loop through blocks
		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


			//for each item in parallel, we check the global counters to determine which hash is submitted
			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


			//offsets is the # of items pulled by shortcutting
			int remaining = buffer_sizes[global_buffer] - offsets[i];

			for (int j = threadID; j < remaining; j+=32){


				key_type hash = buffers[global_buffer][j+offsets[i]];

				//uint64_t  = get_alt_hash(hash, global_buffer);


				int alt_bucket = get_alt_bucket_from_key(hash, i) % BLOCKS_PER_THREAD_BLOCK;
				//int alt_bucket = get_alt_bucket_from_key(hash, global_buffer) % BLOCKS_PER_THREAD_BLOCK;


				if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;


				#if DEBUG_ASSERTS

				assert(j + offsets[i] < buffer_sizes[global_buffer]);

				assert(alt_bucket < BLOCKS_PER_THREAD_BLOCK);
				assert(i < BLOCKS_PER_THREAD_BLOCK);
				assert(alt_bucket != i);


				#endif

				//replace with faster atomic

				//
				//if 	(atomicAdd(&counters[i], (int) 0) < atomicAdd(&counters[alt_bucket], (int) 0)){
				if (atomicCAS(&counters[i], 0, 0) < atomicCAS(&counters[alt_bucket],0,0)){

					slot = atomicAdd(&counters[i], 1);

					//These adds aren't undone on failure as no one else can succeed.
					if (slot < block_type::max_size()){

						//slot - offset = fill+#writes - this is guaranteed to be a free slot
						slot -= offsets[i];

						#if DEBUG_ASSERTS

						//this kind of overwrite is ok when deleting...
						//since we compress and just write out the compressed version
						//with the correct count we are essentially overwriting useless data.
						if (local_blocks->internal_blocks[i].tags[slot].get_key() != 0){
							printf("Overwrite!\n");
						}

						#endif

						local_blocks->internal_blocks[i].tags[slot] = hash;
					


						#if DEBUG_ASSERTS

						assert(slot + offsets[i]  < block_type::max_size());

						#endif

					} else {

						//atomicSub(&counters[i],1);

						//atomicadd fails, try alternate spot
						slot = atomicAdd(&counters[alt_bucket], 1);

						if (slot < block_type::max_size()){

							slot -= offsets[alt_bucket];


							#if DEBUG_ASSERTS

							if (local_blocks->internal_blocks[alt_bucket].tags[slot].get_key() != 0){
								printf("Overwrite!\n");
							}

							#endif


							local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;

							local_blocks->internal_blocks[alt_bucket].tags[slot].mark_secondary();

							#if DEBUG_ASSERTS

							assert(slot + offsets[alt_bucket] < block_type::max_size());

							#endif					

						} else {

							//atomicSub(&counters[alt_bucket],1);

							atomicAdd((unsigned long long int *) misses, 1ULL);

						}



					}


				} else {

					//alt < main slot
					slot = atomicAdd(&counters[alt_bucket], 1);

					if (slot < block_type::max_size()){

						//slot = atomicAdd(&start_counters[alt_bucket], 1);
						slot -= offsets[alt_bucket];

						//temp_tags[alt_bucket*block_type::max_size()+slot] = hash & 0xff;

						#if DEBUG_ASSERTS

						if (local_blocks->internal_blocks[alt_bucket].tags[slot].get_key() != 0){
							printf("Overwrite!\n");
						}

						#endif


						local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;

						local_blocks->internal_blocks[alt_bucket].tags[slot].mark_secondary();


						#if DEBUG_ASSERTS

						assert(slot + offsets[alt_bucket] < block_type::max_size());

						#endif		

					} else {

						//atomicSub(&counters[alt_bucket], 1); 

						//primary insert failed, attempt secondary
						slot = atomicAdd(&counters[i], 1);

						if (slot < block_type::max_size()){

							slot -= offsets[i];

							//temp_tags[i*block_type::max_size()+slot] = hash & 0xff;


							#if DEBUG_ASSERTS

							if (local_blocks->internal_blocks[i].tags[slot].get_key() != 0){
								printf("Overwrite!\n");
							}

							#endif


							local_blocks->internal_blocks[i].tags[slot] = hash;



						


							#if DEBUG_ASSERTS

							assert(slot + offsets[i]  < block_type::max_size());

							#endif

						} else {


							//atomicSub(&counters[alt_bucket], 1);
							atomicAdd((unsigned long long int *) misses, 1ULL);


							}

						}




				}


			}

		}


		//end of for loop

		__syncthreads();

		//no overwrite in above statement.


		//start of dump


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			if (counters[i] >  block_type::max_size()){

				counters[i] =  block_type::max_size();

			}

			#if DEBUG_ASSERTS
			
			assert(counters[i] <=  block_type::max_size());

			#endif


			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			int local_block_offset = block_counters[global_buffer];

			
			int length = counters[i] - offsets[i] - local_block_offset;
	 


			#if DEBUG_ASSERTS


			if (length + local_block_offset + offsets[i] >  block_type::max_size()){

				printf("length violation\n");

				assert(length + local_block_offset + offsets[i] <=  block_type::max_size());

			}
		

			if (! (counters[i] <=  block_type::max_size())){

					printf("Counter size violation\n");
					//start_counters[i] -1
					assert(counters[i] <=  block_type::max_size());

			}

		

			#endif



			// if (length > 32 && threadID == 0)

			// 		insertion_sort_max(&temp_tags[i* block_type::max_size()], length);

			// 	sorting_network_8_bit(&temp_tags[i* block_type::max_size()], length, threadID);

			// 	__syncwarp();

			// 	#if DEBUG_ASSERTS

			// 	assert(short_byte_assert_sorted(&temp_tags[i* block_type::max_size()], length));

			// 	#endif


			//EOD HERE - patch sorting network for 16 bit 


			int tag_fill = local_block_offset;


			templated_insertion_sort<Key, Val, Wrapper>(&local_blocks->internal_blocks[i].tags[tag_fill], length, threadID);

			__syncwarp();


			#if DEBUG_ASSERTS

				if (!assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length)){
						printf("Sort failed For insertion_sort!\n");
				}

			#endif

			//start of 16 bit

			// if (length <= 32){

			// 	#if DEBUG_ASSERTS

			// 		assert(tag_fill + length <= block_type::max_size());

			// 	#endif


			// 	sorting_network<Key, Val, Wrapper>(&local_blocks->internal_blocks[i].tags[tag_fill], length, threadID);

			// 	__syncwarp();


			// 	#if DEBUG_ASSERTS

			// 	//TODO PATCH SORTING NETWORK

			// 	if (!assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length)){
			// 		printf("Sort failed!\n");
			// 	}
			// 	assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			// 	#endif

			// } else {


			// 	#if DEBUG_ASSERTS

			// 		assert(tag_fill + length <= block_type::max_size());

			// 	#endif


			// 	if (threadID ==0)

			// 	insertion_sort<key_type>(&local_blocks->internal_blocks[i].tags[tag_fill], length);

		

			// 	__syncwarp();

			// 	sorting_network(&local_blocks->internal_blocks[i].tags[tag_fill], 32, threadID);

			// 	__syncwarp();


			// 	#if DEBUG_ASSERTS


			// 	assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], 32));

			// 	assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			// 	#endif

			// }

			//end of 16 bit

			assert(length + tag_fill + offsets[i] <=  block_type::max_size());



			//now all three arrays are sorted, and we have a valid target for write-out



			//local_blocks->internal_blocks[i].sorted_bulk_finish(&temp_tags[i* block_type::max_size()+length], &temp_tags[i* block_type::max_size()], length, warpID, threadID);




			//and merge into main arrays
			//uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


			//buffers to be dumped
			//global_buffer -> counter starts at 0, runs to offets[i];
			//temp_tags, starts at 0, runs to get_fill();
			//other temp_tags, starts at get_fill(), runs to length; :D


			#if DEBUG_ASSERTS

			assert(assert_sorted(buffers[global_buffer], offsets[i]));

			assert(local_block_offset == tag_fill);

			//assert(local_block_offset =)

			assert(tag_fill == block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]);


			if (! assert_sorted(&local_blocks->internal_blocks[i].tags[0], tag_fill)){

				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[0], tag_fill));

			}

			assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			assert(blockID*BLOCKS_PER_THREAD_BLOCK +i == global_buffer);






			#endif


			blocks[blockID].internal_blocks[i].dump_all_buffers_sorted(buffers[global_buffer], offsets[i], &local_blocks->internal_blocks[i].tags[0], tag_fill, &local_blocks->internal_blocks[i].tags[tag_fill], length, warpID, threadID, dividing_line);

			block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] = offsets[i] + tag_fill + length;

			//double triple check that dump_all_buffers increments the internal counts like it needs to.


			//maybe this is the magic?

			// #if DEBUG_ASSERTS
			// __threadfence();


			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) != offsets[i]){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) == offsets[i]);

			// }

			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) != tag_fill){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) == tag_fill);
		

			// }


			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) != length ){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) == length);

			// }


			// //assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], buffer_sizes[global_buffer]) == buffer_sizes[global_buffer]);

			
			// #endif

		} //end of 648 - warpID +=32




		#if DEBUG_ASSERTS


		__threadfence();



		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			assert(assert_sorted(blocks[blockID].internal_blocks[i].tags,block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]));

		}
		//let everyone do all checks
		// for (int i =0; i < BLOCKS_PER_THREAD_BLOCK; i+=1){

		// 	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


		// 	for (int j = 0; j < buffer_sizes[global_buffer]; j++){

		// 		assert(query_single_item_sorted_debug(threadID, buffers[global_buffer][j]));
		// 	}

		// }

		#endif



		//end of dump



	}

__device__ void dump_all_buffers_sorted_cycles(thread_team_block<block_type> * local_blocks, int * offsets, int * counters, uint64_t blockID, int warpID, int threadID, uint64_t * misses, uint64_t * cycles){


		uint64_t select_clock_start = clock();

		if (threadID == 0){

			for (int i =warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


				//remaining counters now takes into account the main list as well as new inserts

				counters[i] = offsets[i] + block_counters[blockID*BLOCKS_PER_THREAD_BLOCK + i];

				//local_block_offset;

				//start_counters[i] = 0;

			}

		}

		__syncthreads();


		#if DEBUG_ASSERTS

		for (int i = 0; i < BLOCKS_PER_THREAD_BLOCK; i++){

			assert(counters[i] <= block_type::max_size());
		}

		__syncthreads();

		#endif


		int slot;

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


			//for each item in parallel, we check the global counters to determine which hash is submitted
			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;

			int remaining = buffer_sizes[global_buffer] - offsets[i];

			for (int j = threadID; j < remaining; j+=32){


				key_type hash = buffers[global_buffer][j+offsets[i]];

				//uint64_t  = get_alt_hash(hash, global_buffer);

				int alt_bucket = get_alt_bucket_from_key(hash, i) % BLOCKS_PER_THREAD_BLOCK;
				//int alt_bucket = get_alt_bucket_from_key(hash, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

				if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;


				#if DEBUG_ASSERTS

				assert(j + offsets[i] < buffer_sizes[global_buffer]);

				assert(alt_bucket < BLOCKS_PER_THREAD_BLOCK);
				assert(i < BLOCKS_PER_THREAD_BLOCK);
				assert(alt_bucket != i);


				#endif

				//replace with faster atomic

				//
				//if 	(atomicAdd(&counters[i], (int) 0) < atomicAdd(&counters[alt_bucket], (int) 0)){
				if (atomicCAS(&counters[i], 0, 0) < atomicCAS(&counters[alt_bucket],0,0)){

					slot = atomicAdd(&counters[i], 1);

					//These adds aren't undone on failure as no one else can succeed.
					if (slot < block_type::max_size()){

						//slot - offset = fill+#writes - this is guaranteed to be a free slot
						slot -= offsets[i];

						local_blocks->internal_blocks[i].tags[slot] = hash;
					


						#if DEBUG_ASSERTS

						assert(slot + offsets[i]  < block_type::max_size());

						#endif

					} else {

						//atomicSub(&counters[i],1);

						//atomicadd fails, try alternate spot
						slot = atomicAdd(&counters[alt_bucket], 1);

						if (slot < block_type::max_size()){

							slot -= offsets[alt_bucket];


							local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;

							#if DEBUG_ASSERTS

							assert(slot + offsets[alt_bucket] < block_type::max_size());

							#endif					

						} else {

							//atomicSub(&counters[alt_bucket],1);

							atomicAdd((unsigned long long int *) misses, 1ULL);

						}



					}


				} else {

					//alt < main slot
					slot = atomicAdd(&counters[alt_bucket], 1);

					if (slot < block_type::max_size()){

						//slot = atomicAdd(&start_counters[alt_bucket], 1);
						slot -= offsets[alt_bucket];

						//temp_tags[alt_bucket*block_type::max_size()+slot] = hash & 0xff;


						local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;

						#if DEBUG_ASSERTS

						assert(slot + offsets[alt_bucket] < block_type::max_size());

						#endif		

					} else {

						//atomicSub(&counters[alt_bucket], 1); 

						//primary insert failed, attempt secondary
						slot = atomicAdd(&counters[i], 1);

						if (slot < block_type::max_size()){

							slot -= offsets[i];

							//temp_tags[i*block_type::max_size()+slot] = hash & 0xff;


							local_blocks->internal_blocks[i].tags[slot] = hash;



						


							#if DEBUG_ASSERTS

							assert(slot + offsets[i]  < block_type::max_size());

							#endif

						} else {


							//atomicSub(&counters[alt_bucket], 1);
							atomicAdd((unsigned long long int *) misses, 1ULL);


							}

						}




				}


			}

		}


		//end of for loop

		__syncthreads();

		uint64_t select_clock_end = clock();


		if (threadID == 0){
			atomicAdd((unsigned long long int *) &cycles[7], select_clock_end-select_clock_start);

		}


		//start of dump


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			if (counters[i] >  block_type::max_size()){

				counters[i] =  block_type::max_size();

			}

			#if DEBUG_ASSERTS

			if (counters[i] >  block_type::max_size()){

				counters[i] =  block_type::max_size();

				assert(counters[i] <=  block_type::max_size());
			}
			

			#endif


			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			int local_block_offset = block_counters[global_buffer];

			
			int length = counters[i] - offsets[i] - local_block_offset;
	 


			#if DEBUG_ASSERTS

			if (length + local_block_offset + offsets[i] >  block_type::max_size()){

				assert(length + local_block_offset + offsets[i] <=  block_type::max_size());

			}
		

			if (! (counters[i] <=  block_type::max_size())){

					//start_counters[i] -1
					assert(counters[i] <=  block_type::max_size());

			}

		

			#endif



			// if (length > 32 && threadID == 0)

			// 		insertion_sort_max(&temp_tags[i* block_type::max_size()], length);

			// 	sorting_network_8_bit(&temp_tags[i* block_type::max_size()], length, threadID);

			// 	__syncwarp();

			// 	#if DEBUG_ASSERTS

			// 	assert(short_byte_assert_sorted(&temp_tags[i* block_type::max_size()], length));

			// 	#endif


			//EOD HERE - patch sorting network for 16 bit 


			int tag_fill = local_block_offset;


			//start of 16 bit

			uint64_t sort_clock_start = clock();

			if (length <= 32){

				#if DEBUG_ASSERTS

					assert(tag_fill + length <= block_type::max_size());

				#endif


				sorting_network<Key, Val, Wrapper>(&local_blocks->internal_blocks[i].tags[tag_fill], length, threadID);

				__syncwarp();


				#if DEBUG_ASSERTS

				//TODO PATCH SORTING NETWORK


				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

				#endif

			} else {


				#if DEBUG_ASSERTS

					assert(tag_fill + length <= block_type::max_size());

				#endif


				if (threadID ==0)

				insertion_sort<key_type>(&local_blocks->internal_blocks[i].tags[tag_fill], length);

		

				__syncwarp();

				sorting_network(&local_blocks->internal_blocks[i].tags[tag_fill], 32, threadID);

				__syncwarp();


				#if DEBUG_ASSERTS


				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], 32));

				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

				#endif

			}

			//end of 16 bit

			uint64_t sort_clock_end = clock();


			if (threadID == 0){
			atomicAdd((unsigned long long int *) &cycles[1], sort_clock_end-sort_clock_start);
			}

			#if DEBUG_ASSERTS

			assert(length + tag_fill + offsets[i] <=  block_type::max_size());

			#endif



			//now all three arrays are sorted, and we have a valid target for write-out



			//local_blocks->internal_blocks[i].sorted_bulk_finish(&temp_tags[i* block_type::max_size()+length], &temp_tags[i* block_type::max_size()], length, warpID, threadID);




			//and merge into main arrays
			//uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


			//buffers to be dumped
			//global_buffer -> counter starts at 0, runs to offets[i];
			//temp_tags, starts at 0, runs to get_fill();
			//other temp_tags, starts at get_fill(), runs to length; :D


			#if DEBUG_ASSERTS

			assert(assert_sorted(buffers[global_buffer], offsets[i]));

			assert(local_block_offset == tag_fill);

			//assert(local_block_offset =)

			assert(tag_fill == block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]);


			if (! assert_sorted(&local_blocks->internal_blocks[i].tags[0], tag_fill)){

				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[0], tag_fill));

			}

			assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			assert(blockID*BLOCKS_PER_THREAD_BLOCK +i == global_buffer);






			#endif


			uint64_t dump_clock_start = clock();

			//atomicAdd((unsigned long long int *) &cycles[1], sort_clock_end-sort_clock_start);


			blocks[blockID].internal_blocks[i].dump_all_buffers_sorted(buffers[global_buffer], offsets[i], &local_blocks->internal_blocks[i].tags[0], tag_fill, &local_blocks->internal_blocks[i].tags[tag_fill], length, warpID, threadID, dividing_line);

			block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] = offsets[i] + tag_fill + length;


			uint64_t dump_clock_end = clock();

			if (threadID == 0){
				atomicAdd((unsigned long long int *) &cycles[2], dump_clock_end-dump_clock_start);

			}
			//double triple check that dump_all_buffers increments the internal counts like it needs to.


			//maybe this is the magic?

			// #if DEBUG_ASSERTS
			// __threadfence();


			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) != offsets[i]){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) == offsets[i]);

			// }

			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) != tag_fill){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) == tag_fill);
		

			// }


			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) != length ){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) == length);

			// }


			// //assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], buffer_sizes[global_buffer]) == buffer_sizes[global_buffer]);

			
			// #endif

		} //end of 648 - warpID +=32




		#if DEBUG_ASSERTS


		__threadfence();



		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			assert(assert_sorted(blocks[blockID].internal_blocks[i].tags,block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]));

		}
		//let everyone do all checks
		// for (int i =0; i < BLOCKS_PER_THREAD_BLOCK; i+=1){

		// 	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


		// 	for (int j = 0; j < buffer_sizes[global_buffer]; j++){

		// 		assert(query_single_item_sorted_debug(threadID, buffers[global_buffer][j]));
		// 	}

		// }

		#endif



		//end of dump



	}


__device__ void dump_all_buffers_into_local_block(thread_team_block<block_type> * local_blocks, thread_team_block<block_type> * output_block, int * local_block_counters, int * offsets, int * counters, uint64_t blockID, int warpID, int threadID, uint64_t * misses){



		if (threadID == 0){

			for (int i =warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


				//remaining counters now takes into account the main list as well as new inserts

				counters[i] = offsets[i] + local_block_counters[i];

				//local_block_offset;

				//start_counters[i] = 0;

			}

		}

		__syncthreads();


		#if DEBUG_ASSERTS

		for (int i = 0; i < BLOCKS_PER_THREAD_BLOCK; i++){

			assert(counters[i] <= block_type::max_size());
		}

		__syncthreads();

		#endif


		int slot;

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


			//for each item in parallel, we check the global counters to determine which hash is submitted
			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;

			int remaining = buffer_sizes[global_buffer] - offsets[i];

			for (int j = threadID; j < remaining; j+=32){


				key_type hash = buffers[global_buffer][j+offsets[i]];

				//uint64_t  = get_alt_hash(hash, global_buffer);

				int alt_bucket = get_alt_bucket_from_key(hash, i) % BLOCKS_PER_THREAD_BLOCK;
				//int alt_bucket = get_alt_bucket_from_key(hash, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

				if (alt_bucket == i) alt_bucket = (alt_bucket + 1) % BLOCKS_PER_THREAD_BLOCK;


				#if DEBUG_ASSERTS

				assert(j + offsets[i] < buffer_sizes[global_buffer]);

				assert(alt_bucket < BLOCKS_PER_THREAD_BLOCK);
				assert(i < BLOCKS_PER_THREAD_BLOCK);
				assert(alt_bucket != i);


				#endif

				//replace with faster atomic

				//
				//if 	(atomicAdd(&counters[i], (int) 0) < atomicAdd(&counters[alt_bucket], (int) 0)){
				if (atomicCAS(&counters[i], 0, 0) < atomicCAS(&counters[alt_bucket],0,0)){

					slot = atomicAdd(&counters[i], 1);

					//These adds aren't undone on failure as no one else can succeed.
					if (slot < block_type::max_size()){

						//slot - offset = fill+#writes - this is guaranteed to be a free slot
						slot -= offsets[i];

						local_blocks->internal_blocks[i].tags[slot] = hash;
					


						#if DEBUG_ASSERTS

						assert(slot + offsets[i]  < block_type::max_size());

						#endif

					} else {

						//atomicSub(&counters[i],1);

						//atomicadd fails, try alternate spot
						slot = atomicAdd(&counters[alt_bucket], 1);

						if (slot < block_type::max_size()){

							slot -= offsets[alt_bucket];


							local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;

							#if DEBUG_ASSERTS

							assert(slot + offsets[alt_bucket] < block_type::max_size());

							#endif					

						} else {

							//atomicSub(&counters[alt_bucket],1);

							atomicAdd((unsigned long long int *) misses, 1ULL);

						}



					}


				} else {

					//alt < main slot
					slot = atomicAdd(&counters[alt_bucket], 1);

					if (slot < block_type::max_size()){

						//slot = atomicAdd(&start_counters[alt_bucket], 1);
						slot -= offsets[alt_bucket];

						//temp_tags[alt_bucket*block_type::max_size()+slot] = hash & 0xff;


						local_blocks->internal_blocks[alt_bucket].tags[slot] = hash;

						#if DEBUG_ASSERTS

						assert(slot + offsets[alt_bucket] < block_type::max_size());

						#endif		

					} else {

						//atomicSub(&counters[alt_bucket], 1); 

						//primary insert failed, attempt secondary
						slot = atomicAdd(&counters[i], 1);

						if (slot < block_type::max_size()){

							slot -= offsets[i];

							//temp_tags[i*block_type::max_size()+slot] = hash & 0xff;


							local_blocks->internal_blocks[i].tags[slot] = hash;



						


							#if DEBUG_ASSERTS

							assert(slot + offsets[i]  < block_type::max_size());

							#endif

						} else {


							//atomicSub(&counters[alt_bucket], 1);
							atomicAdd((unsigned long long int *) misses, 1ULL);


							}

						}




				}


			}

		}


		//end of for loop

		__syncthreads();




		//start of dump


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			if (counters[i] >  block_type::max_size()){

				counters[i] =  block_type::max_size();

			}

			#if DEBUG_ASSERTS

			if (counters[i] >  block_type::max_size()){

				counters[i] =  block_type::max_size();

				assert(counters[i] <=  block_type::max_size());
			}
			

			#endif


			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			int local_block_offset = local_block_counters[i];

			
			int length = counters[i] - offsets[i] - local_block_offset;
	 


			#if DEBUG_ASSERTS

			if (length + local_block_offset + offsets[i] >  block_type::max_size()){

				assert(length + local_block_offset + offsets[i] <=  block_type::max_size());

			}
		

			if (! (counters[i] <=  block_type::max_size())){

					//start_counters[i] -1
					assert(counters[i] <=  block_type::max_size());

			}

		

			#endif



			// if (length > 32 && threadID == 0)

			// 		insertion_sort_max(&temp_tags[i* block_type::max_size()], length);

			// 	sorting_network_8_bit(&temp_tags[i* block_type::max_size()], length, threadID);

			// 	__syncwarp();

			// 	#if DEBUG_ASSERTS

			// 	assert(short_byte_assert_sorted(&temp_tags[i* block_type::max_size()], length));

			// 	#endif


			//EOD HERE - patch sorting network for 16 bit 


			int tag_fill = local_block_offset;


			//start of 16 bit

			if (length <= 32){

				#if DEBUG_ASSERTS

					assert(tag_fill + length <= block_type::max_size());

				#endif


				sorting_network<Key, Val, Wrapper>(&local_blocks->internal_blocks[i].tags[tag_fill], length, threadID);

				__syncwarp();


				#if DEBUG_ASSERTS

				//TODO PATCH SORTING NETWORK


				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

				#endif

			} else {


				#if DEBUG_ASSERTS

					assert(tag_fill > 0)

					assert(tag_fill + length <= block_type::max_size());

				#endif


				if (threadID ==0)

				insertion_sort<key_type>(&local_blocks->internal_blocks[i].tags[tag_fill], length);

		

				__syncwarp();

				sorting_network(&local_blocks->internal_blocks[i].tags[tag_fill], 32, threadID);

				__syncwarp();


				#if DEBUG_ASSERTS


				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], 32));

				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

				#endif

			}

			//end of 16 bit

			assert(length + tag_fill + offsets[i] <=  block_type::max_size());



			//now all three arrays are sorted, and we have a valid target for write-out



			//local_blocks->internal_blocks[i].sorted_bulk_finish(&temp_tags[i* block_type::max_size()+length], &temp_tags[i* block_type::max_size()], length, warpID, threadID);




			//and merge into main arrays
			//uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


			//buffers to be dumped
			//global_buffer -> counter starts at 0, runs to offets[i];
			//temp_tags, starts at 0, runs to get_fill();
			//other temp_tags, starts at get_fill(), runs to length; :D


			#if DEBUG_ASSERTS

			assert(assert_sorted(buffers[global_buffer], offsets[i]));

			assert(local_block_offset == tag_fill);

			//assert(local_block_offset =)

			assert(tag_fill == local_block_counters[i]);


			if (! assert_sorted(&local_blocks->internal_blocks[i].tags[0], tag_fill)){

				assert(assert_sorted(&local_blocks->internal_blocks[i].tags[0], tag_fill));

			}

			assert(assert_sorted(&local_blocks->internal_blocks[i].tags[tag_fill], length));

			assert(blockID*BLOCKS_PER_THREAD_BLOCK +i == global_buffer);






			#endif


			output_block->internal_blocks[i].dump_all_buffers_sorted(buffers[global_buffer], offsets[i], &local_blocks->internal_blocks[i].tags[0], tag_fill, &local_blocks->internal_blocks[i].tags[tag_fill], length, warpID, threadID, dividing_line);

			local_block_counters[i] = offsets[i] + tag_fill + length;

			//double triple check that dump_all_buffers increments the internal counts like it needs to.


			//maybe this is the magic?

			// #if DEBUG_ASSERTS
			// __threadfence();


			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) != offsets[i]){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], offsets[i]) == offsets[i]);

			// }

			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) != tag_fill){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[0], tag_fill) == tag_fill);
		

			// }


			// if (blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) != length ){

			// 	assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found_short(threadID, &local_blocks->internal_blocks[i].tags[tag_fill], length) == length);

			// }


			// //assert(blocks[blockID].internal_blocks[i].sorted_bulk_query_num_found(threadID, buffers[global_buffer], buffer_sizes[global_buffer]) == buffer_sizes[global_buffer]);

			
			// #endif

		} //end of 648 - warpID +=32




		#if DEBUG_ASSERTS


		__threadfence();



		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			assert(assert_sorted(output_block->internal_blocks[i].tags,local_block_counters[i]));

		}
		//let everyone do all checks
		// for (int i =0; i < BLOCKS_PER_THREAD_BLOCK; i+=1){

		// 	uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK + i;


		// 	for (int j = 0; j < buffer_sizes[global_buffer]; j++){

		// 		assert(query_single_item_sorted_debug(threadID, buffers[global_buffer][j]));
		// 	}

		// }

		#endif



		//end of dump



	}



	//Queries
	__host__ void bulk_query(bool * hits, uint64_t ext_num_teams){


		bulk_sorted_query_kernel<bulk_tcf<Key, Val, Wrapper>><<<ext_num_teams, BLOCK_SIZE>>>(this, hits);



	}

	__host__ void check_correctness(uint64_t * items, uint64_t nitems){

		uint64_t * final_bucket_ids;

		cudaMalloc((void **)&final_bucket_ids, nitems*sizeof(uint64_t));

		key_type * tags;

		cudaMalloc((void **)&tags, nitems*sizeof(key_type));

		int * alt_bucket;

		cudaMalloc((void **)&alt_bucket, nitems*sizeof(int));

		setup_correctness_kernel<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems-1)/512+1, 512>>>(this, items, final_bucket_ids, tags, alt_bucket, nitems);

		cudaDeviceSynchronize();

		check_correctness_kernel<bulk_tcf<Key, Val, Wrapper>, key_type><<<(nitems-1)/512+1, 512>>>(this, final_bucket_ids, tags, alt_bucket, nitems);

		cudaDeviceSynchronize();

		cudaFree(final_bucket_ids);

		cudaFree(tags);

		cudaFree(alt_bucket);



	}


	__host__ void bulk_delete(bool * hits, uint64_t ext_num_teams){


		bulk_sorted_delete_kernel<bulk_tcf<Key, Val, Wrapper>><<<ext_num_teams, BLOCK_SIZE>>>(this, hits);



	}


	__host__ void get_fill(uint64_t * counter, uint64_t num_blocks){

		
		bulk_get_fill_kernel<bulk_tcf<Key, Val, Wrapper>><<<(num_blocks-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(this, counter, num_blocks);
		
	}


	__device__ bool mini_filter_bulk_deletes(bool * hits){

		__shared__ thread_team_block<block_type> block;

		//uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

		uint64_t blockID = blockIdx.x;

		int warpID = threadIdx.x / 32;

		int threadID = threadIdx.x % 32;


		if (blockID >= num_teams) return false;



		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			block.internal_blocks[i] = blocks[blockID].internal_blocks[i]; 
			//printf("i: %d\n",i);
			
		}

		//keys coming in are identical. - this is expected.

		//therefore...
		//either first delete removes items it should not remove.
		//*or* insert did not properly insert.

		//separate these pulls so that the queries can be encapsulated

		__syncthreads();

		#if DEBUG_ASSERTS

		//debug code
		//first off, all tags should be findable
		for (int i = warpID; i< BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;

			if (threadID == 0){

				for (int j = 0; j < buffer_sizes[global_buffer]; j+=1){

						key_type item = buffers[global_buffer][j];

						int alt_bucket = get_alt_bucket_from_key(item, i) % BLOCKS_PER_THREAD_BLOCK;
						//int alt_bucket = get_alt_bucket_from_key(item, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

						if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

						bool primary_found = block.internal_blocks[i].query_search_linear(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]);
							

						item.mark_secondary();
						bool alt_found = block.internal_blocks[alt_bucket].query_search_linear(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket]);

						if (!(primary_found || alt_found)){
							printf("Bug in delete query\n");
						}

						// if (primary_found && alt_found){
						// 	printf("Potential issue\n");
						// }

				}
			}
		}

		__syncthreads();

		#endif

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			//global buffer blockID*BLOCKS_PER_THREAD_BLOCK + i

			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;

			block.internal_blocks[i].sorted_bulk_delete(block_counters[global_buffer], threadID, buffers[global_buffer], hits_ptr, buffer_sizes[global_buffer]);

			// if (threadID == 0){

			// 	for (int j = 0; j < buffer_sizes[global_buffer]; j += 1){

			// 		key_type item = buffers[global_buffer][j];

			// 		hits_ptr[j] = block.internal_blocks[i].individual_delete(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]);

			// 	}

			// }	

		}


		//stop double write bug
		__syncthreads();


		#if DEBUG_ASSERTS

		//anyone remaining should only be in secondary bucket?


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;

			if (threadID == 0){

				for (int j = 0; j < buffer_sizes[global_buffer]; j+=1){

						if (hits_ptr[j]) continue;

						//printf("Entering check\n");

						key_type item = buffers[global_buffer][j];

						//int alt_bucket = get_alt_bucket_from_key(item, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

						int alt_bucket = get_alt_bucket_from_key(item, i) % BLOCKS_PER_THREAD_BLOCK;

						if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

						

						bool primary_found = block.internal_blocks[i].query_search_linear(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]);
						
						item.mark_secondary();
						bool alt_found = block.internal_blocks[alt_bucket].query_search_linear(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket]);

						//I wanted to be deleted and index wanted to delete me, why did this fail?
						if (primary_found){
							printf("Issue in search, primary not properly deleted.\n");
						}


						if (!(primary_found || alt_found)){
							printf("Not found in either after first delete\n");
						}

				}
			}


		}



		__syncthreads();

		#endif


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;

			if (threadID == 0){

				for (int j = 0; j < buffer_sizes[global_buffer]; j+=1){

					if (!hits_ptr[j]){

						key_type item = buffers[global_buffer][j];

						int alt_bucket = get_alt_bucket_from_key(item, i) % BLOCKS_PER_THREAD_BLOCK;
						//int alt_bucket = get_alt_bucket_from_key(item, global_buffer) % BLOCKS_PER_THREAD_BLOCK;


						item.mark_secondary();


						if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

						hits_ptr[j] = block.internal_blocks[alt_bucket].individual_delete(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket]);
					}

				}


			}

			// for (int j = threadID; j < buffer_sizes[global_buffer]; j+=32){

			// 	if (!hits_ptr[j]){

			// 		key_type item = buffers[global_buffer][j];

			// 		int alt_bucket = get_alt_bucket_from_key(item, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

			// 		if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

			// 		hits_ptr[j] = block.internal_blocks[alt_bucket].individual_delete(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket]);
			// 	}

			// }
		}


		//finally, compress and write out.
		__syncthreads();

		#if DEBUG_ASSERTS

		//second set of queries

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){


			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;

			if (threadID == 0){

				for (int j = 0; j < buffer_sizes[global_buffer]; j+=1){

						if (hits_ptr[j]) continue;

						//printf("Entering check\n");

						key_type item = buffers[global_buffer][j];

						//int alt_bucket = get_alt_bucket_from_key(item, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

						int alt_bucket = get_alt_bucket_from_key(item, i) % BLOCKS_PER_THREAD_BLOCK;

						if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

						bool primary_found = block.internal_blocks[i].query_search_linear(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i]);
						

						item.mark_secondary();
						bool alt_found = block.internal_blocks[alt_bucket].query_search_linear(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket]);

						if (!(primary_found || alt_found)){
							printf("After query, not found in either\n");
						}

						if (primary_found && alt_found){
							printf("Found in both after deletes?\n");
						}

				}
			}


		}


		__syncthreads();

		#endif

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			//this version does not yet support updating the counts.
			int new_counter = blocks[blockID].internal_blocks[i].dump_buffer_compress(block.internal_blocks[i].tags, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i], warpID, threadID, dividing_line);

			if (threadID == 31){

				//printf("Old is %d ,new is %d\n", block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i], new_counter);

				// if (block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] != new_counter){
				// 	printf("Deleting existing 0\n");
				// }
				block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i] = new_counter;
			}


			#if DEBUG_ASSERTS
			if(!assert_sorted(blocks[blockID].internal_blocks[i].tags, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+i])){
				printf("Bug, not sorted after delete\n");
			}

			#endif


		}

		return true;

	}




	__device__ bool mini_filter_bulk_queries(bool * hits){

		__shared__ thread_team_block<block_type> block;

		//uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

		uint64_t blockID = blockIdx.x;

		int warpID = threadIdx.x / 32;

		int threadID = threadIdx.x % 32;


		if (blockID >= num_teams) return false;



		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			block.internal_blocks[i] = blocks[blockID].internal_blocks[i]; 
			//printf("i: %d\n",i);
			
		}

		//separate these pulls so that the queries can be encapsulated

		__syncthreads();

		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			//global buffer blockID*BLOCKS_PER_THREAD_BLOCK + i

			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;

			block.internal_blocks[i].sorted_bulk_query(block_counters[global_buffer], threadID, buffers[global_buffer], hits_ptr, buffer_sizes[global_buffer]);

		}


		for (int i = warpID; i < BLOCKS_PER_THREAD_BLOCK; i+=WARPS_PER_BLOCK){

			uint64_t global_buffer = blockID*BLOCKS_PER_THREAD_BLOCK+i;

			uint64_t global_offset = (buffers[global_buffer] - buffers[0]);

			bool * hits_ptr = hits + global_offset;


			for (int j = threadID; j < buffer_sizes[global_buffer]; j+=32){

				if (!hits_ptr[j]){

					key_type item = buffers[global_buffer][j];



					int alt_bucket = get_alt_bucket_from_key(item, i) % BLOCKS_PER_THREAD_BLOCK;
					//int alt_bucket = get_alt_bucket_from_key(item, global_buffer) % BLOCKS_PER_THREAD_BLOCK;

					item.mark_secondary();

					if (alt_bucket == i) alt_bucket = (alt_bucket +1) % BLOCKS_PER_THREAD_BLOCK;

					hits_ptr[j] = block.internal_blocks[alt_bucket].binary_search_query(item, block_counters[blockID*BLOCKS_PER_THREAD_BLOCK+alt_bucket]);

					// if (!hits_ptr[j]){
					// 	printf("Failed to find!\n");
					// }
				}

			}
		}

		__syncthreads();

		return true;

	}


	__host__ uint64_t get_num_blocks(){


		bulk_tcf<Key, Val, Wrapper> * host_tcf;


		cudaMallocHost((void **)& host_tcf, sizeof(bulk_tcf<Key, Val, Wrapper>));

		cudaMemcpy(host_tcf, this, sizeof(bulk_tcf<Key, Val, Wrapper>), cudaMemcpyDeviceToHost);

		uint64_t blocks_val = host_tcf->num_blocks;

		cudaFreeHost(host_tcf);

		return blocks_val;


	}


	__host__ uint64_t get_num_teams(){


		bulk_tcf<Key, Val, Wrapper> * host_tcf;


		cudaMallocHost((void **)& host_tcf, sizeof(bulk_tcf<Key, Val, Wrapper>));

		cudaMemcpy(host_tcf, this, sizeof(bulk_tcf<Key, Val, Wrapper>), cudaMemcpyDeviceToHost);

		uint64_t teams_val = host_tcf->num_teams;

		cudaFreeHost(host_tcf);


		return teams_val;


	}


	//these  boys are exact!
	//__host__ bulk_insert(key_type * keys);
	

};




template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
__host__ void free_tcf(bulk_tcf<Key, Val, Wrapper> * tcf){


	bulk_tcf<Key, Val, Wrapper> * host_tcf;


	cudaMallocHost((void **)& host_tcf, sizeof(bulk_tcf<Key, Val, Wrapper>));

	cudaMemcpy(host_tcf, tcf, sizeof(bulk_tcf<Key, Val, Wrapper>), cudaMemcpyDeviceToHost);

	cudaFree(tcf);

	cudaFree(host_tcf->blocks);

	cudaFree(host_tcf->block_counters);

	cudaFree(host_tcf->buffers);
	cudaFree(host_tcf->buffer_sizes);

	cudaFreeHost(host_tcf);




}




template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
__host__ bulk_tcf<Key, Val, Wrapper> * build_tcf(uint64_t nitems){


	using key_type = key_val_pair<Key, Val, Wrapper>;

	using block_type = templated_block<key_type>;

	bulk_tcf<Key, Val, Wrapper> * host_tcf;

	cudaMallocHost((void **)&host_tcf, sizeof(bulk_tcf<Key,Val,Wrapper>));

	uint64_t num_teams = (nitems - 1)/(BLOCKS_PER_THREAD_BLOCK*block_type::max_size()) + 1;

	uint64_t num_blocks = num_teams*BLOCKS_PER_THREAD_BLOCK;

	//printf("tcf hash hash %llu thread_team_blocks of %d blocks, total %llu blocks\n", num_teams, BLOCKS_PER_THREAD_BLOCK, num_blocks);
	//printf("Each block is %llu items of size %d, total size %d\n", block_type::max_size(), sizeof(key_type), block_type::max_size()*sizeof(key_type));


	host_tcf->num_teams = num_teams;
	host_tcf->num_blocks = num_blocks;


	int * counters;

	cudaMalloc((void **)&counters, num_blocks*sizeof(int));

	cudaMemset(counters, 0, num_blocks*sizeof(int));

	host_tcf->block_counters = counters;


	thread_team_block<block_type> * blocks;

	cudaMalloc((void **)& blocks, num_teams*sizeof(thread_team_block<block_type>));

	cudaMemset(blocks, 0, num_teams*sizeof(thread_team_block<block_type>));

	host_tcf->blocks = blocks;

	//this should 
	host_tcf->dividing_line = (1ULL << (8*sizeof(Key)-5));
	//buffers

	//printf("dividing_line: %llu\n", host_tcf->dividing_line);

	key_type ** buffers;

	cudaMalloc((void **)&buffers, num_blocks*sizeof(key_type **));

	host_tcf->buffers = buffers;

	int * buffer_sizes;

	cudaMalloc((void **)&buffer_sizes, num_blocks*sizeof(int));

	host_tcf->buffer_sizes = buffer_sizes;


	bulk_tcf<Key, Val, Wrapper> * dev_tcf;


	cudaMalloc((void **)& dev_tcf, sizeof(bulk_tcf<Key, Val, Wrapper>));

	cudaMemcpy(dev_tcf, host_tcf, sizeof(bulk_tcf<Key, Val, Wrapper>), cudaMemcpyHostToDevice);

	cudaFreeHost(host_tcf);



	return dev_tcf;

}


#endif //GPU_BLOCK_
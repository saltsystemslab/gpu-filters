#ifndef XOR_P2_BUCKETIZED 
#define XOR_P2_BUCKETIZED


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace insert_schemes {

#define TCF_SHORTCUT_CUTOFF .6


//The insert schemes are in charge of device side inserts
//they handle memory allocations for the host-side table
// and most of the logic involved in inserting/querying


//Power of N Hashing
//Given a probing scheme with depth N, this insert strategy
//queries the fill of all N buckets

//TODO = separate bucket size from cuda size - use static assert to enforce correctness
// bucket_size/NUM_BUCKETS

//TODO - get godbolt to compile a cmake project? that would be nice.
template <typename Key, typename Val, std::size_t Partition_Size, std::size_t Bucket_Size, template <typename, typename, size_t, size_t> class Internal_Rep, std::size_t Max_Probes, template <typename, std::size_t> class Hasher, template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme>

//template <typename Key, std::size_t Partition_Size, template <typename, std::size_t> class Hasher, std::size_t Max_Probes>
//template <typename Hasher1, typename Hasher2, std::size_t Max_Probes>
struct __attribute__ ((__packed__)) power_two_xor_buckets {


	//tag bits change based on the #of bytes allocated per block
private:




	//These need to be flipped
	Internal_Rep<Key, Val, Partition_Size, Bucket_Size> * slots;

	const uint64_t num_buckets;
	const uint64_t seed;




public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	using probing_scheme_type = Probing_Scheme<Key,Partition_Size, Hasher, Max_Probes>;
	using my_type = power_two_xor_buckets<Key, Val, Partition_Size, Bucket_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme>;

	using rep_type = Internal_Rep<Key, Val, Partition_Size, Bucket_Size>;

	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	//define default constructor so cuda doesn't yell
	__host__ __device__ power_two_xor_buckets(): num_buckets(0), seed(0) {};


	//only allowed to be defined on CPU
	__host__ power_two_xor_buckets(rep_type * ext_slots, uint64_t ext_nslots, uint64_t ext_seed): num_buckets(ext_nslots), seed(ext_seed){
		
		slots = ext_slots;
	}


	__host__ static my_type * generate_on_device(uint64_t ext_nslots, uint64_t ext_seed){

		rep_type * ext_slots;

		uint64_t min_buckets = (ext_nslots-1)/Bucket_Size+1;

		uint64_t true_nslots = min_buckets;

		#if DEBUG_PRINTS

		printf("Constructing table wtih %llu buckets, %llu slots\n", true_nslots, true_nslots*Bucket_Size);
		printf("Using %llu bytes, %llu bytes per item\n", true_nslots*sizeof(rep_type), sizeof(rep_type));

		#endif


		cudaMalloc((void **)& ext_slots, true_nslots*sizeof(rep_type));
		cudaMemset(ext_slots, 0, true_nslots*sizeof(rep_type));

		my_type host_version (ext_slots, min_buckets, ext_seed);

		my_type * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, &host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		return dev_version;



	}

	__host__ static void free_on_device(my_type * dev_version){


		my_type host_version;

		cudaMemcpy(&host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaFree(host_version.slots);

		cudaFree(dev_version);

		return;

	}

	//Given a bucketID, attempt an insert
	// This simplifies the logic of the insert schemes
	// Hopefully without affecting performance?
	__device__ __inline__ bool insert_into_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, uint64_t insert_slot){



       		//insert_slot = insert_slot*Bucket_Size;// + insert_tile.thread_rank();

       		//printf("checking_for_slot\n");


 			return slots[insert_slot].insert(insert_tile, key, val);

	}

	__device__ __inline__ bool insert_delete_into_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, uint64_t insert_slot){



       		//insert_slot = insert_slot*Bucket_Size;// + insert_tile.thread_rank();

       		//printf("checking_for_slot\n");


 			return slots[insert_slot].insert_delete(insert_tile, key, val);

	}

	__device__ __inline__ bool query_into_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val & ext_val, uint64_t insert_slot){


			//if (insert_tile.thread_rank() == 0) printf("In query!\n");

			return slots[insert_slot].query(insert_tile, key, ext_val);

	}

	__device__ __inline__ int check_fill_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, uint64_t insert_slot){


		return slots[insert_slot].get_fill(insert_tile);





	}

	__device__ __inline__ int check_empty_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, uint64_t insert_slot){


		return slots[insert_slot].get_empty(insert_tile);





	}

	__device__ __inline__ bool remove_from_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, uint64_t insert_slot){


			//if (insert_tile.thread_rank() == 0) printf("In query!\n");



     		return slots[insert_slot].remove(insert_tile, key);

	}


	//primary has 0
	__device__ inline Key get_tag_primary(uint64_t hash){


		Key tag = rep_type::tag(hash);

		return tag | (1ULL << (rep_type::tag_bits()-1));

	}

	//secondary has 1.
	__device__ inline Key get_tag_secondary(uint64_t hash){


		uint64_t mask = ~(1ULL << (rep_type::tag_bits()-1));

		return rep_type::tag(hash) & mask;

	}


	//TODO - tag generation needs to come after parity bit.
	__device__ __inline__ bool insert(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		//first step is to init probing scheme

		//if(insert_tile.thread_rank() == 0) printf("Inside of power of n insert\n")

		Hasher<Key, Partition_Size> hasher;

		hasher.init(seed);


		uint64_t full_hash = hasher.hash(key);

		Key primary_tag = get_tag_primary(full_hash);

		Key secondary_tag = get_tag_secondary(full_hash);


		//block index in the vqf
		uint64_t remainder = full_hash >> (rep_type::tag_bits());


		//debug check for correctness
		// if (( (remainder << (rep_type::tag_bits()) ) | tag) != full_hash){
		// 	printf("Bug in getting remainder insert\n");
		// }


		uint64_t block_index = remainder % num_buckets;
		//uint64_t alt_block_index = ((full_hash ^ (tag *  0x5bd1e995)) >> (rep_type::tag_bits())) % num_buckets;

		uint64_t alt_block_index = ((block_index ^ (primary_tag *  0x5bd1e995))) % num_buckets;


		int main_fill = check_empty_bucket(insert_tile, primary_tag, val, block_index);

		// if (main_fill < TCF_SHORTCUT_CUTOFF*Bucket_Size){

		// 	if (insert_into_bucket(insert_tile, primary_tag, val, block_index)) return true;

		// 	return insert_into_bucket(insert_tile, secondary_tag, val, alt_block_index);

		// }



		


		//shortcutting will go here


		//
		//int main_fill = check_empty_bucket(insert_tile, primary_tag, val, block_index);
		int alt_fill = check_empty_bucket(insert_tile, secondary_tag, val, alt_block_index);


		if (main_fill < alt_fill){

			if (insert_into_bucket(insert_tile, primary_tag, val, block_index)) return true;


			return insert_into_bucket(insert_tile, secondary_tag, val, alt_block_index);

		} else {


			if (insert_into_bucket(insert_tile, secondary_tag, val, alt_block_index)) return true;

			return insert_into_bucket(insert_tile, primary_tag, val, block_index);

		}

		//printf("This shouldn't be called.\n");

     	return false;

	}


	__device__ uint64_t get_primary_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key){


		Hasher<Key, Partition_Size> hasher;


		hasher.init(seed);


		uint64_t full_hash = hasher.hash(key);

		Key tag = rep_type::tag(full_hash);


		//block index in the vqf
		uint64_t remainder = full_hash >> (rep_type::tag_bits());


		return remainder % num_buckets;


	}


	__device__ Key get_tag(cg::thread_block_tile<Partition_Size> insert_tile, Key key){


		Hasher<Key, Partition_Size> hasher;

		hasher.init(seed);


		uint64_t full_hash = hasher.hash(key);

		Key tag = rep_type::tag(full_hash);


		return get_tag_primary(full_hash);


	}


	__device__ __inline__ bool insert_with_delete(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		Hasher<Key, Partition_Size> hasher;

		hasher.init(seed);


		uint64_t full_hash = hasher.hash(key);

		Key primary_tag = get_tag_primary(full_hash);

		Key secondary_tag = get_tag_secondary(full_hash);


		//block index in the vqf
		uint64_t remainder = full_hash >> (rep_type::tag_bits());


		//debug check for correctness
		// if (( (remainder << (rep_type::tag_bits()) ) | tag) != full_hash){
		// 	printf("Bug in getting remainder insert delete\n");
		// }


		uint64_t block_index = remainder % num_buckets;
		//uint64_t alt_block_index = ((full_hash ^ (tag *  0x5bd1e995)) >> (rep_type::tag_bits())) % num_buckets;

		uint64_t alt_block_index = ((block_index ^ (primary_tag *  0x5bd1e995))) % num_buckets;

		//shortcutting will go here

		int main_fill = check_fill_bucket(insert_tile, primary_tag, val, block_index);

		// if (main_fill < TCF_SHORTCUT_CUTOFF*Bucket_Size){

		// 	if (insert_delete_into_bucket(insert_tile, primary_tag, val, block_index)) return true;

		// 	return insert_delete_into_bucket(insert_tile, secondary_tag, val, alt_block_index);

		// }

		
		int alt_fill = check_fill_bucket(insert_tile, secondary_tag, val, alt_block_index);


		if (main_fill < alt_fill){

			if (insert_delete_into_bucket(insert_tile, primary_tag, val, block_index)) return true;


			return insert_delete_into_bucket(insert_tile, secondary_tag, val, alt_block_index);

		} else {


			if (insert_delete_into_bucket(insert_tile, secondary_tag, val, alt_block_index)) return true;

			return insert_delete_into_bucket(insert_tile, primary_tag, val, block_index);

		}

		//printf("This shouldn't be called.\n");


     	return false;

	}


	//TODO: replace this with generic upset operation.
	__device__ __inline__ bool insert_if_not_exists(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, Val & ext_val, bool &found_val){

		//first step is to init probing scheme

		//if(insert_tile.thread_rank() == 0) printf("Inside of power of n insert\n");

		printf("This is deprecated, don't use\n");
     	return false;

	}

	__device__ __inline__ bool insert_if_not_exists_delete(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, Val & ext_val, bool &found_val){

		//first step is to init probing scheme

		printf("This is deprecated, don't use\n");

     	return false;

	}

	__device__ __inline__ bool query(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val& ext_val){


		Hasher<Key, Partition_Size> hasher;

		hasher.init(seed);


		uint64_t full_hash = hasher.hash(key);

		Key primary_tag = get_tag_primary(full_hash);

		Key secondary_tag = get_tag_secondary(full_hash);

		//block index in the vqf
		uint64_t remainder = full_hash >> (rep_type::tag_bits());


		//debug check for correctness
		// if (( (remainder << (rep_type::tag_bits()) ) | tag) != full_hash){
		// 	printf("Bug in getting remainder query\n");
		// }


		uint64_t block_index = remainder % num_buckets;
		//uint64_t alt_block_index = ((full_hash ^ (tag *  0x5bd1e995)) >> (rep_type::tag_bits())) % num_buckets;


		uint64_t alt_block_index = ((block_index ^ (primary_tag *  0x5bd1e995))) % num_buckets;

		//shortcutting will go here
		if (query_into_bucket(insert_tile, primary_tag, ext_val, block_index)) return true;

		return query_into_bucket(insert_tile, secondary_tag, ext_val, alt_block_index);

		//printf("This shouldn't be called.\n");

     	return false;



	}

	__device__ __inline__ bool remove(cg::thread_block_tile<Partition_Size> insert_tile, Key key){


		Hasher<Key, Partition_Size> hasher;

		hasher.init(seed);


		uint64_t full_hash = hasher.hash(key);

		Key primary_tag = get_tag_primary(full_hash);
		Key secondary_tag = get_tag_secondary(full_hash);


		//block index in the vqf
		uint64_t remainder = full_hash >> (rep_type::tag_bits());


		//debug check for correctness
		// if (( (remainder << (rep_type::tag_bits()) ) | tag) != full_hash){
		// 	printf("Bug in getting remainder remove\n");
		// }


		uint64_t block_index = remainder % num_buckets;
		//uint64_t alt_block_index = ((full_hash ^ (tag *  0x5bd1e995)) >> (rep_type::tag_bits())) % num_buckets;

		uint64_t alt_block_index = ((block_index ^ (primary_tag *  0x5bd1e995))) % num_buckets;
		//shortcutting will go here


		if (remove_from_bucket(insert_tile, primary_tag, block_index)) return true;

		return remove_from_bucket(insert_tile, secondary_tag, alt_block_index);



	}

	__device__ uint64_t bytes_in_use(){
		return num_buckets*sizeof(rep_type);
	}

	__host__ uint64_t host_bytes_in_use(){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, this, sizeof(my_type), cudaMemcpyDeviceToHost);

		uint64_t ret_val = host_version->num_buckets*sizeof(rep_type);

		cudaFreeHost(host_version);

		return ret_val;



	}

	__host__ uint64_t host_get_num_buckets(){

		my_type * host_version;

		cudaMallocHost((void **)&host_version, sizeof(my_type));

		cudaMemcpy(host_version, this, sizeof(my_type), cudaMemcpyDeviceToHost);

		uint64_t ret_val = host_version->num_buckets;

		cudaFreeHost(host_version);

		return ret_val;



	}


};

//insert_schecmes
}


//poggers
}


#endif //GPU_BLOCK_
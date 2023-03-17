#ifndef LINEAR_INSERT_BUCKET_SCHEME 
#define LINEAR_INSERT_BUCKET_SCHEME


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace insert_schemes {


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
struct __attribute__ ((__packed__)) linear_insert_bucket_scheme {


	//tag bits change based on the #of bytes allocated per block
private:





	Internal_Rep<Key, Val, Partition_Size, Bucket_Size> * slots;

	const uint64_t num_buckets;
	const uint64_t seed;




public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	using probing_scheme_type = Probing_Scheme<Key,Partition_Size, Hasher, Max_Probes>;
	using my_type = linear_insert_bucket_scheme<Key, Val, Partition_Size, Bucket_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme>;

	using rep_type = Internal_Rep<Key, Val, Partition_Size, Bucket_Size>;

	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	//define default constructor so cuda doesn't yell
	__host__ __device__ linear_insert_bucket_scheme(): num_buckets(0), seed(0) {};


	//only allowed to be defined on CPU
	__host__ linear_insert_bucket_scheme(rep_type * ext_slots, uint64_t ext_nslots, uint64_t ext_seed): num_buckets(ext_nslots), seed(ext_seed){
		
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


	__device__ __inline__ bool remove_from_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, uint64_t insert_slot){


			//if (insert_tile.thread_rank() == 0) printf("In query!\n");



     		return slots[insert_slot].remove(insert_tile, key);

	}

	__device__ __inline__ bool insert(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		//first step is to init probing scheme

		//if(insert_tile.thread_rank() == 0) printf("Inside of power of n insert\n");



		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

       		insert_slot = insert_slot % num_buckets;


       		if (insert_into_bucket(insert_tile, key, val , insert_slot)) return true;

       	}  

     	return false;

	}

	__device__ __inline__ bool insert_with_delete(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		//first step is to init probing scheme

		//if(insert_tile.thread_rank() == 0) printf("Inside of power of n insert\n");



		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

       		insert_slot = insert_slot % num_buckets;


       		if (insert_delete_into_bucket(insert_tile, key, val , insert_slot)) return true;

       	}  

     	return false;

	}



	//TODO: replace this with generic upset operation.
	__device__ __inline__ bool insert_if_not_exists(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, Val & ext_val, bool &found_val){

		//first step is to init probing scheme

		//if(insert_tile.thread_rank() == 0) printf("Inside of power of n insert\n");


		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

			insert_slot = insert_slot % num_buckets;

			if (query_into_bucket(insert_tile, key, ext_val, insert_slot)){

				found_val = true;
				return true;
			}

			//I think this works?
			//same as check_fill but we just ballot on trying to insert.
			if (insert_into_bucket(insert_tile, key, val, insert_slot)){

				found_val = false;
				return true;

			}
		}

		
		return false;


	}


	__device__ __inline__ bool insert_if_not_exists_delete(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, Val & ext_val, bool &found_val){

		//first step is to init probing scheme

		//if(insert_tile.thread_rank() == 0) printf("Inside of power of n insert\n");


		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

			insert_slot = insert_slot % num_buckets;

			if (query_into_bucket(insert_tile, key, ext_val, insert_slot)){

				found_val = true;

				remove_from_bucket(insert_tile, key, insert_slot);

				return true;
			}

			//I think this works?
			//same as check_fill but we just ballot on trying to insert.
			if (insert_into_bucket(insert_tile, key, val, insert_slot)){

				found_val = false;
				return true;

			}
		}

		
		return false;


	}

	__device__ __inline__ bool remove(cg::thread_block_tile<Partition_Size> insert_tile, Key key){

		//first step is to init probing scheme

		//if (insert_tile.thread_rank() == 0) printf("Starting outer query!\n");


		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

       			
       		insert_slot = insert_slot % num_buckets;


       		

			if (remove_from_bucket(insert_tile, key, insert_slot)){

				//if (insert_tile.thread_rank() == 0) printf("Found in %llu!\n", insert_slot);
				return true;
			}

			Val fake_val = 0;

			if (check_fill_bucket(insert_tile, key, fake_val, insert_slot) != Bucket_Size){
				return false;
			}
     	

		}

		//if (insert_tile.thread_rank() == 0) printf("Could not find %d\n", key);

		return false;


	}

	__device__ __inline__ bool query(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val& ext_val){

		//first step is to init probing scheme

		//if (insert_tile.thread_rank() == 0) printf("Starting outer query!\n");


		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next(rep_type::tag(key))){

       			
       		insert_slot = insert_slot % num_buckets;


       		

			if (query_into_bucket(insert_tile, key, ext_val, insert_slot)){

				//if (insert_tile.thread_rank() == 0) printf("Found in %llu!\n", insert_slot);
				return true;
			}

			if (check_fill_bucket(insert_tile, key, ext_val, insert_slot) != Bucket_Size){
				return false;
			}
     	
     	

		}

		//if (insert_tile.thread_rank() == 0) printf("Could not find %d\n", key);

		return false;


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
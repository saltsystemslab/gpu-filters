#ifndef POGGERS_TCF
#define POGGERS_TCF

#include <hipex/hipex.hpp>

#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>

#include <poggers/probing_schemes/linear_probing.cuh>

// new container for 2-byte key val pairs

#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/dynamic_container.cuh>



#include <poggers/sizing/default_sizing.cuh>

#include <poggers/representations/packed_bucket.cuh>

#include <poggers/insert_schemes/linear_insert_buckets.cuh>
#include <poggers/tables/base_table.cuh>

#include <poggers/tables/bucketed_table.cuh>


#include <poggers/representations/grouped_storage_sub_bits.cuh>

#include <poggers/probing_schemes/xor_power_of_two.cuh>

#include <poggers/probing_schemes/double_hashing.cuh>

#include <poggers/insert_schemes/xor_p2_buckets.cuh>


namespace poggers {

namespace data_structs {

    template<typename Key_type, typename Val_type, std::size_t key_bits, std::size_t val_bits, std::size_t CG_Size, std::size_t Bucket_Size> 
    struct tcf_wrapper {




    	using backing_type = poggers::tables::bucketed_table<
    	Key_type, Val_type,
		poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		CG_Size, Bucket_Size, poggers::insert_schemes::linear_insert_bucket_scheme, 10000000ULL, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher>;


		// using backing_type = poggers::tables::bucketed_table<
    	// Key_type, Val_type,
		// poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		// CG_Size, Bucket_Size, poggers::insert_schemes::linear_insert_bucket_scheme_parity, 16, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher>;


		// using backing_type = poggers::tables::bucketed_table<
    	// Key_type, Val_type,
		// poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		// CG_Size, Bucket_Size, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::XORPowerOfTwoHasher, poggers::hashers::murmurHasher>;


		using tcf = poggers::tables::bucketed_table<
    	Key_type, Val_type,
		poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		CG_Size, Bucket_Size, poggers::insert_schemes::power_two_xor_buckets, 2, poggers::probing_schemes::XORPowerOfTwoHasher, poggers::hashers::murmurHasher, true, backing_type>;


		
		// using tcf = poggers::tables::bucketed_table<
    	// Key_type, Val_type,
		// poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		// CG_Size, Bucket_Size, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::XORPowerOfTwoHasher, poggers::hashers::murmurHasher>;


		//big buckets

		// using backing_tcf = poggers::tables::bucketed_table<
    	// Key_type, Val_type,
		// poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		// CG_Size, Bucket_Size, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::XORPowerOfTwoHasher, poggers::hashers::murmurHasher>;


		// using tcf = poggers::tables::bucketed_table<
    	// Key_type, Val_type,
		// poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		// CG_Size, Bucket_Size, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::XORPowerOfTwoHasher, poggers::hashers::murmurHasher, true, backing_tcf>;


		// using tcf = poggers::tables::bucketed_table<
    	// Key_type, Val_type,
		// poggers::representations::dynamic_bucket_container<poggers::representations::bit_grouped_container<key_bits, val_bits>::representation>::representation, 
		// CG_Size, Bucket_Size, poggers::insert_schemes::linear_insert_bucket_scheme, 100000000ULL, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher>;



		static tcf * generate_on_device(uint64_t nitems, uint64_t seed){

			uint64_t backing_size = (nitems/100);

			if (backing_size == 0) backing_size +=10;

			//printf("Backing table is %llu\n", backing_size);

			auto sizing = poggers::sizing::size_in_num_slots<2>(nitems, backing_size);

			//auto sizing = poggers::sizing::size_in_num_slots<1>(nitems);

			return tcf::generate_on_device(&sizing, seed);

		}

		static tcf * free_on_device(tcf * ext_filter){

			tcf::free_on_device(ext_filter);
		}


	};



}


}


#endif //Poggers TCF guard
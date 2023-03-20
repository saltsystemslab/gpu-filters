#ifndef TEMPLATED_SORTING 
#define TEMPLATED_SORTING


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "bulk_tcf_metadata.cuh"
#include "bulk_tcf_key_val_pair.cuh"
#include <variant>


#ifndef KEY_VAL_SPLIT
#define KEY_VAL_SPLIT 1
#endif


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

//handled by one thread atm
template <typename Key_type>
__device__ void insertion_sort(Key_type * items, int nitems){


	for (int i = nitems-1; i >= 32; i--){

		Key_type max = 0;

		int max_index = 0;

		//loop through all items beow the established index
		for (int j=0; j <= i; j++){


			if (items[j] > max){

				max = items[j];

				max_index = j;
			}


		}


		//swap i and max_index

		//max is already set to items[j], save a cycle
		items[max_index] = items[i];
		items[i] = max;





	}


}

template <typename Key, typename Val, template<typename T> typename Wrapper>
__device__ void templated_insertion_sort(key_val_pair<Key, Val, Wrapper> * items, int nitems, int warpID){

	if (warpID != 0) return;

	key_val_pair<Key, Val, Wrapper> storage;

	for (int i = 0; i < nitems; i++){

		auto min = items[i].get_key();

		for (int j = i; j < nitems; j++){

			if (items[j].get_key() < min){

				min = items[j].get_key();

				storage = items[i];

				items[i] = items[j];

				items[j] = storage;


			}

		}

	}


}

template <typename Key, typename Val, template<typename T> typename Wrapper>
__device__ void sorting_network(key_val_pair<Key, Val, Wrapper> * items, int nitems, int warpID){


	//this implementation uses batcher odd-even mergesort


	//implementation 2 - massively parallel radix sort

	// uint8_t * main_buffer = items;

	// uint8_t * alt_buffer = items + 64;


	//calculate bit
	for (int mask_bit = 0; mask_bit < 8ULL *sizeof(Key); mask_bit++){

		Key mask = ((Key) 1) << mask_bit;


		bool my_bit = false;


		if (warpID < nitems){

			my_bit = ((items[warpID].get_key() & mask) > (Key) 0);
		}



		


		//and ballot synch on this bit
		unsigned int result = __ballot_sync(0xffffffff, my_bit);

		//histogram

		//TODO: double check me - this "histogram" calc should prevent incorrect items from summing

		//int num_ones = __popc(result);

		int num_zeros = nitems - __popc(result);


		//prefix sum of zeros

		int zero_sum = !my_bit;



		for (int i =1; i<=16; i*=2){

			int n = __shfl_up_sync(0xffffffff, zero_sum, i, 32);

			if ((warpID) >= i) zero_sum +=n;

		}

		//subtracting read gives us an initial start
		zero_sum = zero_sum - !my_bit;



		int one_sum = my_bit;

		for (int i =1; i<=16; i*=2){

			int n = __shfl_up_sync(0xffffffff, one_sum, i, 32);

			if ((warpID) >= i) one_sum +=n;

		}

		//subtracting read gives us an initial start
		one_sum = one_sum - my_bit;


		//__syncwarp();


		key_val_pair<Key, Val, Wrapper> my_item;

		if (warpID < nitems){

			my_item = items[warpID];

		}

		__syncwarp();

		if (warpID < nitems){

				

			//if one bit(num_zeros + one_sum);
			//if zero !bit(zero_sum)

			int write_index = zero_sum;

			if (my_bit){

				write_index = num_zeros+one_sum;

			}


			items[ write_index ] = my_item;


		}

		__syncwarp();


		// my_bit = items[warpID] & mask;
		// //and ballot synch on this bit
		// result = __ballot_sync(0xffffffff, my_bit);

		//__syncwarp();



	}




}



#endif //GPU_BLOCK_
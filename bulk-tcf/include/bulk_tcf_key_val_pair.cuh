#ifndef KEY_VAL_PAIR_H 
#define KEY_VAL_PAIR_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "bulk_tcf_metadata.cuh"
#include "bulk_tcf_hashutil.cuh"
#include <variant>


#ifndef KEY_VAL_SPLIT
#define KEY_VAL_SPLIT 1
#endif


// struct __attribute__ ((__packed__)) val_storage {
	
// 	Val val;

// };

template <typename Val>
struct __attribute__ ((__packed__)) empty_wrapper {

	__host__ __device__ empty_wrapper(Val val){}

	__host__ __device__ empty_wrapper(){}

	__host__ __device__ bool has_value(){ return false;}


};


struct __attribute__ ((__packed__)) empty{

};


struct __attribute__ ((__packed__)) test_struct{

	empty val;
};


template <typename Val>
struct __attribute__ ((__packed__)) wrapper{

	Val val;

	__host__ __device__ wrapper(): val(0) {}

	__host__ __device__ wrapper(Val new_val): val(new_val){}

	__host__ __device__ bool has_value(){ return true;}
};

template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper >
struct __attribute__ ((__packed__)) key_val_pair: private Wrapper<Val> {

	private:
		Key key;

	public:

		__host__ __device__ key_val_pair(){

		}

		//constructor
		__host__ __device__ key_val_pair (Key const & key, Val const & val)
		: Wrapper<Val>(val), key(key){}

		__host__ __device__ key_val_pair (Key const & key)
		: key(key){}

		__host__ __device__ Val& get_val(){
			return (Val&)*this;
		}

		__host__ __device__ Key& get_key(){
			return (Key &) this->key;
		}

		__host__ __device__ void set_val(Val new_val){

			((Val *) this)[0] = new_val;
		}

		__host__ __device__ void set_key(Key new_key){

			new_key += (new_key == 0);

			this->key = new_key;
		}

		__host__ __device__ void set_key_empty(){

			this->key = 0;
		}

		__host__ __device__ void mark_primary(){

			Key newkey = this->key;

			Key key_mask = (1ULL << (sizeof(Key)*8-1))-1;

			set_key(newkey & key_mask);

		}

		__host__ __device__ void mark_secondary(){

			Key newkey = this->key;

			Key key_mask = (1ULL << (sizeof(Key)*8-1));

			set_key(newkey | key_mask);

		}

		__host__ __device__ bool reset_key_atomic(Key ext_key){

			return typed_atomic_write(&key, ext_key, Key{0ULL});

		}

		__host__ __device__ void pack_into_pair(Key new_key, Val new_val){

			set_val(new_val);
			set_key(new_key);
		}

		__host__ __device__ bool is_empty(){
			return (get_key() == 0ULL);
		}


};


// template <typename Key, typename Val, template<typename T> typename Wrapper>
// __device__ void pack_into_pair(key_val_pair<Key, Val, Wrapper> & pair, Key & new_key, Val & new_val ){

// 	pair.set_key(new_key);
// 	pair.set_val(new_val);


// }

// template <typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
// struct key_val_pair{


// 	//tag bits change based on the #of bytes allocated per block

// 	storage_pair<Key, Val, Wrapper> internal_storage;

// 	//key_val_pair (Key const)

// 	//you only use this when you have vals
// 	key_val_pair(Key const & key, Val const & val): storage_pair<Key, Val, Wrapper>(key, val){}

// 	key_val_pair(Key const & key): storage_pair<Key, Val, Wrapper>(key){}

// 	key_val_pair(){}

// 	__host__ __device__ Key& get_key(){

// 		return internal_storage.get_key();
// 	}

// 	__host__ __device__ Val& get_val(){

// 		return internal_storage.get_val();
// 	}



// };

// template <typename Key>
// struct key_val_pair<Key, void>{



// };


template <typename Key, typename Val, template<typename T> typename Wrapper>
__host__ __device__ bool operator<(key_val_pair<Key, Val, Wrapper> A, key_val_pair<Key, Val, Wrapper> B){


	return A.get_key() < B.get_key();

}

template <typename Key, typename Val, template<typename T> typename Wrapper>
__host__ __device__ bool operator<=(key_val_pair<Key, Val, Wrapper> A, key_val_pair<Key, Val, Wrapper> B){

	return A.get_key() <= B.get_key();

}

template <typename Key, typename Val, template<typename T> typename Wrapper>
__host__ __device__ bool operator>=(key_val_pair<Key, Val, Wrapper> A, key_val_pair<Key, Val, Wrapper> B){

	return A.get_key() >= B.get_key();

}

template <typename Key, typename Val, template<typename T> typename Wrapper>
__host__ __device__ bool operator==(key_val_pair<Key, Val, Wrapper> A, key_val_pair<Key, Val, Wrapper> B){

	return A.get_key() == B.get_key();

}

template <typename Key, typename Val, template<typename T> typename Wrapper>
__host__ __device__ bool operator!=(key_val_pair<Key, Val, Wrapper> A, key_val_pair<Key, Val, Wrapper> B){

	return A.get_key() != B.get_key();

}

template <typename Key, typename Val, template<typename T> typename Wrapper>
__host__ __device__ bool operator>(key_val_pair<Key, Val, Wrapper> A, key_val_pair<Key, Val, Wrapper> B){

	return A.get_key() > B.get_key();

}


template <typename Key, typename Val, template<typename T> typename Wrapper>
__host__ __device__ Key operator/(key_val_pair<Key, Val, Wrapper >A, uint64_t other){

	return A.get_key() / other;
}



#endif //GPU_BLOCK_
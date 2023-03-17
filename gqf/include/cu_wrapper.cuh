/*
 * ============================================================================
 *
 *        Authors:  Hunter McCoy <hjmccoy@lbl.gov>
 *                  
 *
 * ============================================================================
 */

#ifndef CURAND_WRAPPER_CUH
#define CURAND_WRAPPER_CUH
#endif

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      if ((x) == CURAND_STATUS_NOT_INITIALIZED) printf("Gen no init\n"); \
      }} while(0)


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>

#include "include/zipf.cuh"

struct curand_generator {


	void init(uint64_t seed, int rand_type, uint64_t backing_size);
	void gen_next_batch(uint64_t noutputs);
	void reset_to_defualt();
	__device__ uint64_t get_next(uint64_t tid);
	uint64_t * yield_backing();

	void destroy();

	void setup_host_backing(uint64_t full_size);

private:


	uint64_t * backing;
	uint64_t state;
	uint64_t seed;
	uint64_t backing_size;
	curandGenerator_t gen;
	int type;
	uint64_t * host_backing;
	uint64_t host_counter;
	uint64_t max_host_counter;


};

//STATE TYPES
//state 0: uniform pregen
//state 1: streaming uniform pregen


// //backing size must be large enough to satisfy one full request set
// __global__ void clip_values(uint64_t * items, uint64_t nitems, uint64_t clip){


	
// }


//for now, keep as global-ish entity wrapped in this file
void curand_generator::init(uint64_t inp_seed, int rand_type, uint64_t _backing_size){

	//malloc backing
	seed = inp_seed;
	state = rand_type;

	backing_size = _backing_size;

	//curandGenerator_t temp_gen;

	uint64_t * temp_backing;
	//TODO: remove the managed from this
	cudaMalloc((void **) &temp_backing,backing_size*sizeof(uint64_t));
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));

	//gen = temp_gen;
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1ULL*seed));
	backing = temp_backing;


}


//This generates a zipfian distribution when called iff the generator is set to 2
//all other generators use cuRand so no work is done on host. 
void curand_generator::setup_host_backing(uint64_t full_size){

	if (state != 2){

		//printf("Calling host setup on non-zipfian randomness, ignoring.\n");
		return;
	}


	uint64_t * temp_arr = (uint64_t * ) malloc(full_size * sizeof(uint64_t));

	generate_random_keys(temp_arr, full_size, full_size, 1);

	host_backing = temp_arr;
	host_counter = 0;
	max_host_counter = full_size;
}

void curand_generator::gen_next_batch(uint64_t noutputs){


	if (state==0 || state == 1){

		CURAND_CALL(curandGenerate(gen, (unsigned int *) backing, 2*backing_size));


	} else if (state == 2){

		
		cudaMemcpy(backing, host_backing+host_counter, noutputs, cudaMemcpyHostToDevice);

		host_counter +=noutputs;

		if (host_counter > max_host_counter){
			printf("ERROR Zipfian too small\n");
			abort();
		}

	} else {

		printf("generator not configured for this type yet.\n");
		abort();
	}



}

void curand_generator::reset_to_defualt(){

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));

}

__device__ uint64_t curand_generator::get_next(uint64_t tid){

	return backing[tid];

}

uint64_t * curand_generator::yield_backing(){

	if (state ==0 || state == 2){
		return backing;
	} else if (state == 1){

		uint64_t * temp_backing = backing;
		uint64_t * new_backing;
		cudaMalloc((void **) &new_backing, backing_size*sizeof(uint64_t));
		backing = new_backing;

		return temp_backing;



	} else {

		printf("generator not configured for this type yet.\n");

		return nullptr;
	}
	
}

void curand_generator::destroy(){

	CURAND_CALL(curandDestroyGenerator(gen));
	cudaFree(backing);


}


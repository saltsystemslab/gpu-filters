/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>

#include <openssl/rand.h>

#include "bulk_tcf_host.cuh"




#define COUNTING_CYCLES 0

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void check_hits(bool * hits, uint64_t * misses, uint64_t nitems){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nitems) return;

	if (!hits[tid]){

		atomicAdd((unsigned long long int *) misses, 1ULL);

	}
}

__global__ void check_hits_print_misses(bool * hits, uint64_t * keys, uint64_t * misses, uint64_t nitems){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nitems) return;

	if (!hits[tid]){

		printf("Index %llu: Key %llu - %u missed\n", tid, keys[tid], (uint16_t) keys[tid]);

		atomicAdd((unsigned long long int *) misses, 1ULL);

	}
}

template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> split_insert_timing(host_bulk_tcf<Large_Keys, Key, Val, Wrapper> * my_tcf, Large_Keys * keys, uint64_t nvals, uint64_t * misses){


	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();

	my_tcf->bulk_insert(keys, nvals, misses);

	cudaDeviceSynchronize();
	
	gpuErrchk( cudaPeekAtLastError() );

	auto end = std::chrono::high_resolution_clock::now();

  	std::chrono::duration<double> diff = end-start;

  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	//misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> bulk_query_timing(host_bulk_tcf<Large_Keys, Key, Val, Wrapper> * my_tcf, Large_Keys * large_keys, uint64_t nvals, uint64_t * misses){


	auto start = std::chrono::high_resolution_clock::now();


	auto hits = my_tcf->bulk_query_scrambled(large_keys, nvals);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	//check_hits_print_misses<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, large_keys, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	//misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> bulk_fp_timing(host_bulk_tcf<Large_Keys, Key, Val, Wrapper> * my_tcf, Large_Keys * large_keys, uint64_t nvals, uint64_t * misses){


	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	
	auto hits = my_tcf->bulk_query(large_keys, nvals);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);



	//check hits

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "FP Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("FP Sorted Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu, ratio: %f\n", misses[0], 1.0 * (nvals - misses[0])/nvals);  

  	cudaDeviceSynchronize();

  //	misses[0] = 0;

  	cudaDeviceSynchronize();

   return diff;
}

template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> bulk_delete_timing(host_bulk_tcf<Large_Keys, Key, Val, Wrapper> * my_tcf, Large_Keys * large_keys, uint64_t nvals, uint64_t * misses){



	

	//cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	
	bool * hits = my_tcf->bulk_delete(large_keys, nvals);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits_print_misses<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, large_keys, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Deleted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Bulk Deletes per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	//misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


template <typename T>
__host__ T * generate_data(uint64_t nitems){


	//malloc space

	T * vals = (T *) malloc(nitems * sizeof(T));


	//			   100,000,000
	uint64_t cap = 100000000ULL;

	for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

		uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


		RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



		to_fill += togen;

		printf("Generated %llu/%llu\n", to_fill, nitems);

	}

	return vals;
}

template <typename T>
__host__ T * load_main_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/tcf_data/main_data-32-data.txt";

	//char main_location[] = "/pscratch/sd/h/hunterm/tcf_data/main_data-32-data.txt";

	char * vals = (char * ) malloc(nitems * sizeof(T));

	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(T), pFile);

	if (result != nitems*sizeof(T)) abort();



	// //current supported format is no spacing one endl for the file terminator.
	// if (myfile.is_open()){


	// 	getline(myfile, line);

	// 	strncpy(vals, line.c_str(), sizeof(uint64_t)*nitems);

	// 	myfile.close();
		

	// } else {

	// 	abort();
	// }


	return (T *) vals;


}

template <typename T>
__host__ T * load_alt_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/tcf_data/fp_data-32-data.txt";

	//char main_location[] = "/pscratch/sd/h/hunterm/tcf_data/fp_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(T));


	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(T), pFile);

	if (result != nitems*sizeof(T)) abort();



	return (T *) vals;


}

template <typename Key, typename SmallKey>
__host__ void sawtooth_test(int num_bits, int num_partitions, int num_rounds){


	using Filter = host_bulk_tcf<Key, SmallKey>;

	uint64_t nitems = (1ULL << num_bits);

	uint64_t items_per_partition = (nitems*.8)/num_partitions;


	Key ** inserted_keys = (Key **) malloc(sizeof(Key *)*num_partitions);

   Key ** new_keys = (Key **) malloc(sizeof(Key *)*num_rounds);

   for (int i = 0; i < num_partitions; i++){

      inserted_keys[i] = generate_data<Key>(items_per_partition);

   }

   for (int i =0; i < num_rounds; i++){

      new_keys[i] = generate_data<Key>(items_per_partition);

   }


   Key * dev_keys;

   cudaMalloc((void **)&dev_keys, sizeof(Key)*items_per_partition);

   Filter * tcf = Filter::host_build_tcf(nitems);


   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*6);
   cudaDeviceSynchronize();


   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;
   misses[5] = 0;


   printf("Starting sawtooth test for %llu items, %llu items per %d partitions\n", nitems, items_per_partition, num_partitions);

   cudaDeviceSynchronize();

   for (int i = 0; i < num_partitions; i++){

      printf("Starting setup round %d\n",i);

      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);


      cudaDeviceSynchronize();

      split_insert_timing(tcf, dev_keys, items_per_partition, misses);

      cudaDeviceSynchronize();

   }


   for (int i = 0; i < num_rounds; i++){


      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

     
      cudaDeviceSynchronize();

      bulk_delete_timing(tcf, dev_keys, items_per_partition, misses+1);

      cudaDeviceSynchronize();


      //false negative query
      for (int j = i+1; j < num_partitions; j++){

         cudaMemcpy(dev_keys, inserted_keys[j], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();


      	bulk_query_timing(tcf, dev_keys, items_per_partition, misses+3);


         cudaDeviceSynchronize();
      }


      for (int k = 0; k<i; k++){


         cudaMemcpy(dev_keys, new_keys[k], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();

         bulk_query_timing(tcf, dev_keys, items_per_partition, misses+3);


      }

      //and finally re-add

      cudaDeviceSynchronize();

      cudaMemcpy(dev_keys, new_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);


      split_insert_timing(tcf, dev_keys, items_per_partition, misses+4);

      cudaDeviceSynchronize();
   


   }

   cudaDeviceSynchronize();


   printf("Insert fails: %llu, delete misses: %llu, False negatives: %llu, re-add misses %llu\n", misses[0], misses[1],  misses[3], misses[4]);


   return;




}



int main(int argc, char** argv) {
	

	if (argc < 4) {
      fprintf(stderr, "Please specify the log of the number of items to test with.\n");
      exit(1);

   }

   int nbits = atoi(argv[1]);

   int num_partitions = atoi(argv[2]);

   int num_rounds = atoi(argv[3]);

	sawtooth_test<uint64_t, uint16_t>(nbits, num_partitions, num_rounds);

	return 0;

}

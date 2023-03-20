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
__host__ std::chrono::duration<double> split_insert_timing(host_bulk_tcf<Large_Keys, Key, Val, Wrapper> * my_tcf, Large_Keys * keys, Val * vals, uint64_t nvals, uint64_t * misses){


	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();

	my_tcf->bulk_insert_values(keys, vals, nvals, misses);

	cudaDeviceSynchronize();
	
	gpuErrchk( cudaPeekAtLastError() );

	auto end = std::chrono::high_resolution_clock::now();

  	std::chrono::duration<double> diff = end-start;

  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	printf("New fill: %llu\n", my_tcf->get_fill());

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> bulk_query_timing(host_bulk_tcf<Large_Keys, Key, Val, Wrapper> * my_tcf, Large_Keys * large_keys, Val * vals, uint64_t nvals, uint64_t * misses){


	auto start = std::chrono::high_resolution_clock::now();


	auto hits = my_tcf->bulk_query_values(large_keys, vals, nvals);

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

  	misses[0] = 0;

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

  	misses[0] = 0;

  	cudaDeviceSynchronize();

   return diff;
}

template <typename Large_Keys, typename Key, typename Val = empty, template<typename T> typename Wrapper = empty_wrapper>
__host__ std::chrono::duration<double> bulk_delete_timing(host_bulk_tcf<Large_Keys, Key, Val, Wrapper> * my_tcf, Large_Keys * large_keys, uint64_t nvals, uint64_t * misses){



	

	// cudaMalloc((void **) & hits, nvals*sizeof(bool));

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

  	misses[0] = 0;

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

int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);

	uint64_t num_batches = atoi(argv[2]);

	double batch_percent = 1.0 / num_batches;


	//uint64_t nitems = (1ULL << nbits) * .85;
	uint64_t nitems = (1ULL << nbits) * .8;


	//add one? just to guarantee that the clip is correct
	uint64_t items_per_batch = 1.05*nitems * batch_percent;


	printf("Starting test with %d bits, %llu items inserted in %d batches of %d.\n", nbits, nitems, num_batches, items_per_batch);


	//swap this to remainder type
	using key_type = uint16_t;
	using val_type = uint16_t;
	using main_data_type = key_val_pair<key_type, val_type, wrapper>;

	using tcf_type = host_bulk_tcf<uint64_t, key_type, val_type, wrapper>;

	uint64_t * keys;
	uint64_t * dev_keys;


	//main_data_type * vals;

	//comment this out

	//start up keys as uint
	// keys = (uint64_t *) malloc(nitems*sizeof(uint64_t));

	// for (uint64_t i=0; i< nitems; i++){
	// 	keys[i] = i;
	// }
	keys = generate_data<uint64_t>(nitems);

	//keys = load_main_data<main_data_type>(nitems);


	uint64_t * fp_keys;

	//main_data_type * fp_vals;

	//generate fp data to see comparison with true inserts
	fp_keys = generate_data<uint64_t>(nitems);





	cudaMalloc((void ** )& dev_keys, items_per_batch*sizeof(uint64_t));


	val_type * vals = generate_data<uint16_t>(nitems);

	val_type * dev_vals;

	cudaMalloc((void **)&dev_vals, items_per_batch*sizeof(uint16_t));

	val_type * returned_vals;

	cudaMallocHost((void **)&returned_vals, items_per_batch*sizeof(uint16_t));



	//allocate misses counter
	uint64_t * misses;
	cudaMallocManaged((void **)& misses, sizeof(uint64_t));

	misses[0] = 0;

	#if COUNTING_CYCLES

	uint64_t * cycles;

	cudaMallocManaged((void**)&cycles, 8*sizeof(uint64_t));

	cycles[0] = 0;
	cycles[1] = 0;
	cycles[2] = 0;
	cycles[3] = 0;
	cycles[4] = 0;
	cycles[5] = 0;
	cycles[6] = 0;
	cycles[7] = 0;


	uint64_t * num_warps;

	cudaMallocManaged((void**)&num_warps, 6*sizeof(double));

	cudaDeviceSynchronize();


	#endif


	//change the way tcf is built to better suit test and use cases? TODO with active reconstruction for exact values / struct support

	//quad_hash_table * ht =  build_hash_table(1ULL << nbits);

	tcf_type * tcf = tcf_type::host_build_tcf(1ULL << nbits);
	//bulk_tcf<key_type> * tcf = build_tcf<key_type>( (uint64_t)(1ULL << nbits));

	//std::chrono::duration<double> diff = std::chrono::nanoseconds::zero();


	std::chrono::duration<double>  * insert_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
	std::chrono::duration<double>  * query_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
	std::chrono::duration<double>  * fp_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
	std::chrono::duration<double>  * delete_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
	
	uint64_t * batch_amount = (uint64_t *) malloc(num_batches*sizeof(uint64_t));





	printf("Setup done\n");

	//wipe_vals<<<nitems/32+1, 32>>>(dev_vals, nitems);


	cudaDeviceSynchronize();
	

	for (int batch = 0; batch< num_batches; batch++){

		//calculate size of segment

		printf("Batch %d:\n", batch);



		//runs from batch/num_batches*nitems to batch
		uint64_t start = batch*nitems/num_batches;
		uint64_t end = (batch+1)*nitems/num_batches;
		if (end > nitems) end = nitems;

		uint64_t items_to_insert = end-start;


		assert(items_to_insert <= items_per_batch);

		batch_amount[batch] = items_to_insert;

		printf("First 10 vals: ");

		for (int i =0; i< 10; i++){
			printf("%u ", vals[i+start]);
		}

		printf("\n");

		//prep dev_vals for this round

		cudaMemcpy(dev_keys, keys + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		//this passes
		//tcf->check_correctness(dev_keys, items_to_insert);

		cudaDeviceSynchronize();


		cudaMemcpy(dev_keys, keys + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(val_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		//launch inserts


		insert_diff[batch] = split_insert_timing<uint64_t, key_type, val_type, wrapper>(tcf, dev_keys, dev_vals, items_to_insert, misses);


		
		cudaDeviceSynchronize();



		cudaMemcpy(dev_keys, keys + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(val_type), cudaMemcpyHostToDevice);


		cudaDeviceSynchronize();


		//launch queries
		query_diff[batch] = bulk_query_timing(tcf, dev_keys, dev_vals, items_to_insert, misses);


		cudaDeviceSynchronize();


		cudaMemcpy(returned_vals, dev_vals, sizeof(val_type)*items_to_insert, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		printf("First 10 vals: ");

		for (int i =0; i< 10; i++){
			printf("%u ", returned_vals[i+start]);
		}

		printf("\n");



		cudaMemcpy(dev_keys, fp_keys + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		//cudaMemcpy(dev_vals, fp_vals + start, items_to_insert*sizeof(main_data_type), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


		//false queries
		fp_diff[batch] = bulk_fp_timing(tcf, dev_keys, items_to_insert, misses);


		cudaDeviceSynchronize();



		cudaMemcpy(dev_keys, keys + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);


		cudaDeviceSynchronize();


		//launch queries
		delete_diff[batch] = bulk_delete_timing(tcf, dev_keys, items_to_insert/2, misses);


		cudaDeviceSynchronize();

		cudaMemcpy(dev_keys, keys + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
 	
 		//printf("Deleted\n");
		// bulk_query_timing(tcf, dev_keys, items_to_insert/2, misses);

		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(val_type), cudaMemcpyHostToDevice);


		bulk_query_timing(tcf, dev_keys, dev_vals, items_to_insert, misses);


		// cudaMemcpy(dev_keys, keys + start, items_to_insert*sizeof(uint64_t), cudaMemcpyHostToDevice);

		// cudaDeviceSynchronize();
 		
 		// printf("Not deleted\n");
		// bulk_query_timing(tcf, dev_keys+items_to_insert/2, items_to_insert/2, misses);



		//keep some organized spacing
		printf("\n\n");

		fflush(stdout);

		cudaDeviceSynchronize();



	}

	std::chrono::duration<double> summed_insert_diff = std::chrono::nanoseconds::zero();

	for (int i =0; i < num_batches;i++){
		summed_insert_diff += insert_diff[i];
	}

	std::chrono::duration<double> summed_query_diff = std::chrono::nanoseconds::zero();

	for (int i =0; i < num_batches;i++){
		summed_query_diff += query_diff[i];
	}

	std::chrono::duration<double> summed_fp_diff = std::chrono::nanoseconds::zero();

	for (int i =0; i < num_batches;i++){
		summed_fp_diff += fp_diff[i];
	}

	std::chrono::duration<double> summed_delete_diff = std::chrono::nanoseconds::zero();

	for (int i =0; i < num_batches;i++){
		summed_delete_diff += delete_diff[i];
	}

	printf("Tests Finished.\n");

	std::cout << "Queried " << nitems << " in " << summed_insert_diff.count() << " seconds\n";

	printf("Final speed: %f\n", nitems/summed_insert_diff.count());

	if (argc == 4){

		printf("Dumping into file\n");

		const char * dir = "batched_results/";

		char filename_insert[256];
		char filename_lookup[256];
		char filename_false_lookup[256];
		char filename_delete[256];
		char filename_aggregate[256];

		const char * insert_op = "_insert_";

		snprintf(filename_insert, strlen(dir) + strlen(argv[3]) + strlen(insert_op) + strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], insert_op, argv[1], argv[2]);

		const char * lookup_op = "_lookup_";

		snprintf(filename_lookup, strlen(dir) + strlen(argv[3]) + strlen(lookup_op) + strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], lookup_op, argv[1], argv[2]);

		const char * fp_ops = "_fp_";

		snprintf(filename_false_lookup, strlen(dir) + strlen(argv[3]) + strlen(fp_ops) + strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], fp_ops, argv[1], argv[2]);

		const char * delete_op = "_delete_";

		snprintf(filename_delete, strlen(dir) + strlen(argv[3]) + strlen(fp_ops) + strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], delete_op, argv[1], argv[2]);


		const char * agg_ops = "_aggregate_";

		snprintf(filename_aggregate, strlen(dir) + strlen(argv[3]) + strlen(agg_ops)+ strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], agg_ops, argv[1], argv[2]);


		FILE *fp_insert = fopen(filename_insert, "w");
		FILE *fp_lookup = fopen(filename_lookup, "w");
		FILE *fp_false_lookup = fopen(filename_false_lookup, "w");
		FILE *fp_delete = fopen(filename_delete, "w");
		FILE *fp_agg = fopen(filename_aggregate, "w");

		if (fp_insert == NULL) {
			printf("Can't open the data file %s\n", filename_insert);
			exit(1);
		}

		if (fp_lookup == NULL ) {
		    printf("Can't open the data file %s\n", filename_lookup);
			exit(1);
		}

		if (fp_false_lookup == NULL) {
			printf("Can't open the data file %s\n", filename_false_lookup);
			exit(1);
		}

		if (fp_delete == NULL) {
			printf("Can't open the data file %s\n", filename_delete);
			exit(1);
		}

		if (fp_agg == NULL) {
			printf("Can't open the data file %s\n", filename_aggregate);
			exit(1);
		}


		printf("Writing results to file: %s\n",  filename_insert);

		fprintf(fp_insert, "x_0 y_0\n");
		for (int i = 0; i < num_batches; i++){
			fprintf(fp_insert, "%d", i*100/num_batches);

			fprintf(fp_insert, " %f\n", batch_amount[i]/insert_diff[i].count());
		}
		printf("Insert performance written!\n");

		fclose(fp_insert);


		printf("Writing results to file: %s\n",  filename_lookup);

		fprintf(fp_lookup, "x_0 y_0\n");
		for (int i = 0; i < num_batches; i++){
			fprintf(fp_lookup, "%d", i*100/num_batches);

			fprintf(fp_lookup, " %f\n", batch_amount[i]/query_diff[i].count());
		}
		printf("lookup performance written!\n");

		fclose(fp_lookup);



		printf("Writing results to file: %s\n",  filename_false_lookup);

		fprintf(fp_false_lookup, "x_0 y_0\n");
		for (int i = 0; i < num_batches; i++){
			fprintf(fp_false_lookup, "%d", i*100/num_batches);

			fprintf(fp_false_lookup, " %f\n", batch_amount[i]/fp_diff[i].count());
		}
		printf("false_lookup performance written!\n");

		fclose(fp_false_lookup);


		printf("Writing results to file: %s\n",  filename_delete);

		fprintf(fp_delete, "x_0 y_0\n");
		for (int i = 0; i < num_batches; i++){
			fprintf(fp_delete, "%d", i*100/num_batches);

			fprintf(fp_delete, " %f\n", batch_amount[i]/fp_diff[i].count());
		}
		printf("Delete performance written!\n");

		fclose(fp_delete);


		printf("Writing results to file: %s\n",  filename_aggregate);

		//fprintf(fp_agg, "x_0 y_0\n");

		fprintf(fp_agg, "Aggregate inserts: %f\n", nitems/summed_insert_diff.count());
		fprintf(fp_agg, "Aggregate Queries: %f\n", nitems/summed_query_diff.count());
		fprintf(fp_agg, "Aggregate fp: %f\n", nitems/summed_fp_diff.count());



		printf("false_lookup performance written!\n");

		fclose(fp_agg);



	}


	#if COUNTING_CYCLES

	printf("Cycle counts: insert_total, load, distribute, sorting, merging, hashing, set_buffer, set_len\n%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n", cycles[0], cycles[6], cycles[7], cycles[1], cycles[2], cycles[3], cycles[4], cycles[5]);



	printf("%f, %f, %f, %f, %f, %f, %f, %f\n", 1.0*cycles[0]/num_warps[0], 1.0*cycles[6]/num_warps[0], 1.0*cycles[7]/num_warps[0], 1.0*cycles[1]/num_warps[0], 1.0*cycles[2]/num_warps[0], 1.0*cycles[3]/num_warps[1], 1.0*cycles[4]/num_warps[2], 1.0*cycles[5]/num_warps[3]);
	


	// printf("Hashing, set buffer, set len, load, distribute, sorting, merging");
	// printf("%f, %f, %f, %f, %f, %f, %f, %f\n",  )
	cudaFree(cycles);

	cudaFree(num_warps);
	#endif


	free(keys);

	free(fp_keys);

	cudaFree(dev_keys);

	cudaFree(misses);

	//free_tcf(tcf);

	tcf_type::host_free_tcf(tcf);

	
	//free counters	

	return 0;

}

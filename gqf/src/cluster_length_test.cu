/*
 * ============================================================================
 *
 *        Authors:  
 *					Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
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
#include <openssl/rand.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include "include/gqf_int.cuh"
#include "hashutil.cuh"
#include "include/gqf.cuh"
//#include "src/gqf.cu"
#include <fstream>
#include <string>
#include <algorithm>


__global__ void one_thread_gqf_cluster(QF * qf, uint64_t * num_clusters){


	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid != 0) return;

	printf("Thread 0 on the case.\n");


	uint64_t i =0;

	uint64_t start =0;
	uint64_t next_index;



	while (start < qf->metadata->xnslots){

		uint64_t next_index = first_empty_slot_wrapper(qf, start);



		

		

		if (start == next_index){
			start++;
		} else {
			//printf("cluster %llu : %llu -> %llu\n", i, start, next_index);
			start = next_index;
			i++;
		}
		

	}
	

	num_clusters[0] = i;

	
}


__global__ void one_thread_cluster_write(QF * qf, uint64_t * clusters){

	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid != 0) return;


	uint64_t i =0;

	uint64_t start =0;
	uint64_t next_index;



	while (start < qf->metadata->xnslots){

		uint64_t next_index = first_empty_slot_wrapper(qf, start);



		

		

		if (start == next_index){
			start++;
		} else {

			clusters[i] = next_index - start;
			//printf("cluster %llu : %llu -> %llu\n", i, start, next_index);
			start = next_index;
			i++;
		}
		

	}


}


#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "Please specify the log of the number of slots in the CQF.\n");

		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies] filename\n");

		exit(1);

	}
	if (argc < 3){

		fprintf(stderr, "Please specify output filenum\n");
		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies] filename\n");

		exit(1);
	}


	QF qf;
	uint64_t qbits = atoi(argv[1]);
	uint64_t rbits = 8;
	uint64_t nhashbits = qbits + rbits;
	uint64_t nslots = (1ULL << qbits);
	//this can be changed to change the % it fills up
	uint64_t nvals = 95 * nslots / 100;
	//uint64_t nvals =  nslots/2;
	//uint64_t nvals = 4;
	//uint64_t nvals = 1;
	uint64_t* vals;

	uint64_t * nums;
	uint64_t * counts;

	/* Initialise the CQF */
	if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, false, 0)) {
		fprintf(stderr, "Can't allocate CQF.\n");
		abort();
	}


	uint64_t output_num = atoi(argv[2]);

	bool bulk = true;



	//check if pregen
	int preset = 0;


	nums = (uint64_t*)malloc(nvals * sizeof(vals[0]));
	counts = (uint64_t*)malloc(nvals * sizeof(vals[0]));

	vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
	uint64_t i = 0;

	qf_set_auto_resize(&qf, false);

	if (preset == 1){

		printf("Using preset data\n");

		std::fstream preset_data;

		preset_data.open(argv[4]);

		if (preset_data.is_open()){
			std::string tp;
			while(std::getline(preset_data, tp) && i < nvals){

				char * end;
				vals[i] = std::strtoull(tp.c_str(), &end, 10);
				vals[i] = (1 * vals[i]) % qf.metadata->range;
				i++;
			}





		} else {
			printf("Error opening file %s\n", argv[4]);
		}

		preset_data.close();

		if (i < nvals);

		nvals = i;


	


	} else if (preset == 2){




		RAND_bytes((unsigned char*)nums, sizeof(*vals) * nvals);
		RAND_bytes((unsigned char*)counts, sizeof(*vals) * nvals);


		printf("Generated backing data\n");
		uint64_t cap = 10;


		uint64_t i = 0;

		while (i < nvals){


			uint64_t num;
			uint64_t count;
			

			num = (1 * nums[i]) % qf.metadata->range;

			count = (1 * counts[i]) % cap + 1;

			assert(count > 0);

			for (uint64_t j =i; j < i+count; j++){

				if (j < nvals) vals[j] = num;

			}

			i+=count;



		}

		//shuffle vals
		std::random_device rd;
	    std::mt19937 g(rd());
	 
		std::shuffle(vals, vals+nvals, g);

	} else {

		printf("Using regular data\n");

		/* Generate random values */
		
		RAND_bytes((unsigned char*)vals, sizeof(*vals) * nvals);
		//uint64_t* _vals;
		for (uint64_t i = 0; i < nvals; i++) {
		vals[i] = (1 * vals[i]) % qf.metadata->range;
		//vals[i] = hash_64(vals[i], BITMASK(nhashbits));
		}

	}

	

	//copy vals to device

	uint64_t * dev_vals;

	cudaMalloc((void **)&dev_vals, nvals*sizeof(uint64_t));

	cudaMemcpy(dev_vals, vals, nvals*sizeof(uint64_t), cudaMemcpyHostToDevice);

	// vals = (uint64_t *) malloc(nvals * sizeof(uint64_t));
	// for (uint64_t i =0l; i< nvals; i++){
	// 	vals[i] = i;
	// }

	srand(0);
	/* Insert keys in the CQF */
	printf("starting kernel\n");
	//qf_gpu_launch(&qf, vals, nvals, key_count, nhashbits, nslots);

	QF* dev_qf;
	qf_malloc_device(&dev_qf, qbits, true);
	cudaDeviceSynchronize();
	auto start = std::chrono::high_resolution_clock::now();

	if (bulk){	
		bulk_insert(dev_qf, nvals, dev_vals, 0);
	} else {
		bulk_insert_reduce(dev_qf, nvals, dev_vals, 0);
	}
	
	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

	cudaMemcpy(dev_vals, vals, nvals*sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	start = std::chrono::high_resolution_clock::now();

	uint64_t misses = bulk_get_misses_wrapper(dev_qf, dev_vals, nvals);

	cudaDeviceSynchronize();

	end = std::chrono::high_resolution_clock::now();


  	diff = end-start;

	assert(misses == 0);

	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Queries per second: %f\n", nvals/diff.count());



  	cudaDeviceSynchronize();


  	//read from gqf

  	QF * host_qf = (QF *) malloc(sizeof(QF));

  	cudaMemcpy(host_qf, dev_qf, sizeof(QF), cudaMemcpyDeviceToHost);


  	qfmetadata * host_metadata = (qfmetadata *) malloc(sizeof(qfmetadata));

  	cudaMemcpy(host_metadata, host_qf->metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);

  	qfblock * host_blocks = (qfblock *) malloc(sizeof(qfblock) * host_metadata->nblocks);


  	cudaMemcpy(host_blocks, host_qf->blocks, sizeof(qfblock)*host_metadata->nblocks, cudaMemcpyDeviceToHost);


  	uint64_t * num_clusters;

  	cudaMallocManaged((void **)& num_clusters, sizeof(uint64_t));

  	cudaDeviceSynchronize();

  	num_clusters[0] = 0;

  	cudaDeviceSynchronize();

  	host_qf->metadata = host_metadata;
  	host_qf->blocks = host_blocks;



  	one_thread_gqf_cluster<<<1,1>>>(dev_qf, num_clusters);


 // 	std::vector<uint64_t> cluster_lens;

  	cudaDeviceSynchronize();


  	uint64_t * cluster_counts;

  	cudaMallocManaged((void **)&cluster_counts, num_clusters[0]*sizeof(uint64_t));


  	printf("Num clusters seen by host: %llu\n", num_clusters[0]);

  	cudaDeviceSynchronize();

  	one_thread_cluster_write<<<1,1>>>(dev_qf, cluster_counts);


  	cudaDeviceSynchronize();


  	printf("First len: %llu\n", cluster_counts[0]);



  	// uint64_t current_index = 0;
  	// uint64_t next_index;

  	// bool do_continue = true;

  	// while (do_continue){


  	// 	try {
  	// 			next_index = host_debug_find_first_empty_slot(host_qf, current_index+1);


  	// 	if (next_index >= host_qf->metadata->nslots) do_continue = false;
  	// 	break;

  	// 	}

  	// 	catch (const std::exception& e){

  	// 		do_continue = false;
  	// 		break;
  	// 	}
  	

  	// 	if (current_index == next_index){
  	// 		do_continue = false;
  	// 	} else{

  	// 		cluster_lens.push_back(next_index - current_index);
  	// 		current_index = next_index;
  	// 	}
  	// }


  	// printf("Finished main experiment.\n");

  	// printf("Num clusters: %llu\n", cluster_lens.size());

  	// //loop and count the zeros
  	// uint64_t counter = 0;



  	// for (int i =0; i < host_metadata->nblocks; i++){

  	// 	for (int j=0; j < QF_SLOTS_PER_BLOCK; j++){


  	// 		if (host_blocks[i].slots[j] == 0){


  	// 			if (counter != 0){
  	// 				cluster_lens.push_back(counter);
  	// 				counter = 0;
  	// 			}
  				

  	// 		} else {




  	// 			counter += 1;

  	// 		}
  			

  	// 	}
  	// }


  	// //uint64_t output_num = 0;

  	// cluster_lens.push_back(counter);


  	std::ofstream myfile;

  	std::string fname = "results/results_" + std::to_string(output_num) + ".txt";

  	myfile.open(fname);



  	for (uint64_t i =0; i < num_clusters[0]; i++){

  		myfile << cluster_counts[i] << std::endl;

  	}

  	// myfile.close();

  	free(host_qf);

  	free(host_metadata);

  	free(host_blocks);


	qf_destroy_device(dev_qf);

	printf("GPU launch succeeded\n");
	fflush(stdout);


	return 0;

}

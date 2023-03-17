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
#include "include/zipf.cuh"
//#include "src/gqf.cu"
#include <fstream>
#include <string>
#include <algorithm>


#define KMER_SIZE 19
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "Please specify the log of the number of slots in the CQF.\n");

		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies, fastq] filename\n");

		exit(1);

	}
	if (argc < 3){

		fprintf(stderr, "Please specify 'bulk' or 'reduce'\n");
		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies, 3 fastq] filename\n");

		exit(1);
	}

	if (argc < 4) {
		fprintf(stderr, "Please specify random or preset data.\n");
		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies, 3 fastq] filename\n");

		exit(1);

	}

	printf("This is a set of tests to verify GQF performance and correctness.\n");
	printf("Testing against other filters is handled by test.cu.\n");
	

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


	bool bulk = true;

	if (atoi(argv[2]) != 0){

		printf("Using reduce.\n");
		bulk = false;

	}


	//check if pregen
	int preset = atoi(argv[3]);


	nums = (uint64_t*)malloc(nvals * sizeof(vals[0]));
	counts = (uint64_t*)malloc(nvals * sizeof(vals[0]));

	vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
	uint64_t i = 0;

	qf_set_auto_resize(&qf, false);



	if (preset == 3){


		uint64_t kmer_count = 0;


		std::string filename (argv[4]);

		//std::vector<std::string> fastq_strings;

		std::vector<std::string> kmers;

		printf("Loading fastq file %s.\n", argv[4]);

		printf("%s\n", argv[4]);



		std::ifstream fastq_data (argv[4]);


		if (!fastq_data){
			std::cerr << "Cannot open file: " << argv[4] << std::endl;
		}

		//fastq_data.ignore();
		std::string tp;


		bool do_continue = true;

		while(std::getline(fastq_data, tp) && do_continue){

		

			if (tp.find_first_not_of("ACTGN") == std::string::npos){



			

			//std::cout << tp.c_str() << std::endl;
			if (tp.size() > 0){


				for (int j = KMER_SIZE; j < tp.length(); j++){

					kmers.push_back(tp.substr(j-KMER_SIZE, KMER_SIZE).c_str());

					kmer_count += 1;
					if (kmer_count >= nvals){

						do_continue = false;
						break;
					}

				}
			}


				//break into kmers
			}
			//fastq_strings.push_back(tp.c_str());

			
			}
		
		//}

		fastq_data.close();

		// for (int i=0; i < 10; i++){
		// 	printf("%d: %s\n", i, fastq_strings[i].c_str());
		// }
		
		// printf("Total # of reads: %d\n", fastq_strings.size());


		// //split to kmers



		// for (int i =0; i < fastq_strings.size(); i++){

		// 	for (int j = KMER_SIZE; j < fastq_strings[i].length(); j++){

		// 		kmers.push_back(fastq_strings[i].substr(j-KMER_SIZE, KMER_SIZE).c_str());

		// 	}
		// }

		// for (int i=0; i < 10; i++){
		// 	printf("%d: %s\n", i, kmers[i].c_str());
		// }

		printf("# kmers: %llu\n", kmers.size());

		//fastq_strings.clear();

		std::vector<uint64_t> hashes;


		for (int i = 0; i < kmers.size(); i++){
			hashes.push_back(MurmurHash64A(kmers[i].c_str(), KMER_SIZE, 1));
		}
		//uint64_t MurmurHash64A ( const void * key, int len, unsigned int seed )

		for (int i=0; i < 10; i++){
			printf("%d: %llu\n", i, hashes[i]);
		}


		printf("# hashes: %llu\n", hashes.size());

		kmers.clear();

		free(vals);

		nvals = hashes.size();

		vals = (uint64_t * ) malloc (nvals*sizeof(uint64_t));

		memcpy(vals, hashes.data(), nvals*sizeof(uint64_t));

		hashes.clear();

		//return 0;



	} else if (preset == 1){

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
		printf("Generation is done using RAND_bytes and a single CPU thread, expect a long wait while data is generated if nbits > 26.\n");
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

	} else if (preset == 4){

		printf("generating Zipfian data!\n");

		//do we need to free vals here? zipfian should generate new malloc.


		//uint64_t * ext_vals = (uint64_t*)malloc(nvals * sizeof(uint64_t));


		generate_random_keys(vals, nvals, nvals, 1);

		for (uint64_t i=0; i < nvals; i++){

			vals[i] = (1 *vals[i]) % qf.metadata->range;
		}

	} else {

		printf("Using regular data\n");
		printf("Generation is done using RAND_bytes and a single CPU thread, expect a long wait while data is generated if nbits > 26.\n");
	

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




  	printf("Generating clusters, writing to file\n");

  	uint64_t * cluster_lengths;

  	cudaMallocManaged((void**)& cluster_lengths, nvals*sizeof(uint64_t));

    uint64_t * max_cluster;

  	cudaMallocManaged((void**)& max_cluster, 1*sizeof(uint64_t));

  	cudaDeviceSynchronize();




  	find_clusters<<<1,1>>>(dev_qf, cluster_lengths, max_cluster);

  	cudaDeviceSynchronize();


  	FILE *fp_insert = fopen("clusters.txt", "a");


  	for (uint64_t i=0; i < max_cluster[0]; i++){

  		fprintf(fp_insert, " %llu\n", cluster_lengths[i]);

  	}





  	cudaMemcpy(dev_vals, vals, nvals*sizeof(uint64_t), cudaMemcpyHostToDevice);
  	cudaDeviceSynchronize();

  	start = std::chrono::high_resolution_clock::now();


  	//delete a batch of items in parallel
	bulk_delete(dev_qf, nvals, dev_vals, 0);

	cudaDeviceSynchronize();

	end = std::chrono::high_resolution_clock::now();


  	diff = end-start;


	std::cout << "Deleted " << nvals << " in " << diff.count() << " seconds\n";
	printf("Deletes per second: %f\n", nvals/diff.count());

	cudaMemcpy(dev_vals, vals, nvals*sizeof(uint64_t), cudaMemcpyHostToDevice);
  	cudaDeviceSynchronize();

  	misses = bulk_get_misses_wrapper(dev_qf, dev_vals, nvals);

  	cudaDeviceSynchronize();

  	//we should find no items, they have all been deleted.
  	assert(misses == nvals);

	qf_destroy_device(dev_qf);

	printf("GPU launch succeeded\n");
	fflush(stdout);


	return 0;

}

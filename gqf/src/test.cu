/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>  
 *                  Hunter McCoy <hjmccoy@lbl.gov>  
 *
 * ============================================================================
 */

#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <openssl/rand.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <assert.h> 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <chrono>

//include for cuRand generators
#include "include/cu_wrapper.cuh"



#include "include/gqf_wrapper.cuh"
#include "include/point_wrapper.cuh"
#include "include/rsqf_wrapper.cuh"
#include "include/sqf_wrapper.cuh"
#include "include/bloom_wrapper.cuh"

//new
#include "include/gqf_wrapper_cooperative.cuh"
#include "include/point_wrapper_coop.cuh"


#ifndef  USE_MYRANDOM
#define RFUN random
#define RSEED srandom
#else
#define RFUN myrandom
#define RSEED mysrandom



static unsigned int m_z = 1;
static unsigned int m_w = 1;
static void mysrandom (unsigned int seed) {
	m_z = seed;
	m_w = (seed<<16) + (seed >> 16);
}

static long myrandom()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return ((m_z << 16) + m_w) % 0x7FFFFFFF;
}
#endif




//filter ops
typedef int (*init_op)(uint64_t nvals, uint64_t hash, uint64_t buf_size);
typedef int (*destroy_op)();
typedef int (*bulk_insert_op)(uint64_t * vals, uint64_t nvals);
typedef uint64_t (*bulk_find_op)(uint64_t * vals, uint64_t nvals);
typedef void (*bulk_delete_op)(uint64_t * vals, uint64_t nvals);


typedef struct filter {
	init_op init;
	destroy_op destroy;
	bulk_insert_op bulk_insert;
	bulk_find_op bulk_lookup;
	bulk_delete_op bulk_delete;
	bulk_find_op bulk_fp_lookup;

} filter;





filter gqf = {
	gqf_init,
	gqf_destroy,
	gqf_bulk_insert,
	gqf_bulk_get,
	gqf_bulk_delete,
	gqf_bulk_get_fp
};

// filter mhm2_map = {
// 	map_init,
// 	map_destroy,
// 	map_bulk_insert,
// 	map_bulk_get
// };

filter bloom = {
	bloom_init,
	bloom_destroy,
	bloom_bulk_insert,
	bloom_bulk_get,
	bloom_bulk_delete,
	bloom_bulk_get_fp
};

// filter one_bit_bloom = {
// 	one_bit_bloom_init,
// 	one_bit_bloom_insert,
// 	one_bit_bloom_lookup,
// 	one_bit_bloom_range,
// 	one_bit_bloom_destroy,
// 	one_bit_bloom_iterator,
// 	one_bit_bloom_get,
// 	one_bit_bloom_next,
// 	one_bit_bloom_end,
// 	one_bit_bloom_bulk_insert,
// 	one_bit_bloom_prep_vals,
// 	one_bit_bloom_bulk_get,
// 	one_bit_bloom_xnslots
// };

filter point = {
	point_init,
	point_destroy,
	point_bulk_insert_wrapper,
	point_bulk_get,
	point_bulk_delete,
	point_bulk_get_fp
};


filter rsqf = {
	rsqf_init,
	rsqf_destroy,
	rsqf_bulk_insert,
	rsqf_bulk_get,
	rsqf_bulk_delete,
	rsqf_bulk_get_fp
};

filter sqf = {
	sqf_init,
	sqf_destroy,
	sqf_bulk_insert,
	sqf_bulk_get,
	sqf_bulk_delete,
	sqf_bulk_get_fp
};


filter coop_gqf = {
	coop_gqf_init,
	coop_gqf_destroy,
	coop_gqf_bulk_insert,
	coop_gqf_bulk_get,
	coop_gqf_bulk_delete,
	coop_gqf_bulk_get
};


filter coop_point = {
	coop_point_init,
	coop_point_destroy,
	coop_point_bulk_insert_wrapper,
	coop_point_bulk_get,
	coop_point_bulk_delete,
	coop_point_bulk_get
};

uint64_t * zipfian_backing;


uint64_t tv2msec(struct timeval tv)
{
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int cmp_uint64_t(const void *a, const void *b)
{
	const uint64_t *ua = (const uint64_t*)a, *ub = (const uint64_t *)b;
	return *ua < *ub ? -1 : *ua == *ub ? 0 : 1;
}

void usage(char *name)
{
	printf("%s [OPTIONS]\n"
				 "Options are:\n"
				 "  -n nslots     [ log_2 of filter capacity.  Default 22 ]\n"
				 "  -r nruns      [ number of runs.  Default 1 ]\n"
				 "  -p npoints    [ number of points on the graph.  Default 20 ]\n"
				 "  -b buf_size   [ log_2 of buffer capacity, default is nslots/npoints ]\n"
				 "  -d datastruct [ Default bulk] [ bulk | point | bloom | sqf | rsqf ]\n"
				 "  -f outputfile [ Default gqf. ]\n"
				 "  -e            [ Test and record deletions. Disabled by default]\n",
				 name);
}

int main(int argc, char **argv)
{


	uint32_t nbits = 22, nruns = 1;
	unsigned int npoints = 20;
	uint64_t nslots = (1ULL << nbits), nvals = 950*nslots/1000;

	const char *datastruct = "bulk";
	const char *outputfile = "gqf";

	filter filter_ds;

	unsigned int i, j, exp, run;
	struct std::chrono::time_point<std::chrono::high_resolution_clock> tv_insert[100][1];
	struct std::chrono::time_point<std::chrono::high_resolution_clock> tv_exit_lookup[100][1];
	struct std::chrono::time_point<std::chrono::high_resolution_clock> tv_false_lookup[100][1];

	struct std::chrono::time_point<std::chrono::high_resolution_clock> tv_delete[100][1];

	struct std::chrono::duration<int64_t, std::nano> insert_times[100];
	struct std::chrono::duration<int64_t, std::nano> exit_times[100];
	struct std::chrono::duration<int64_t, std::nano> false_times[100];
	struct std::chrono::duration<int64_t, std::nano> delete_times[100];


	bool fp_queries = false;
	bool verbose = false;

	uint64_t fps = 0;
	//default buffer of 20;
	bool bufset = false;
	uint64_t buf_bits = 18;
	uint64_t buf_size = (1ULL << 20);


	#ifndef __x86_64

	printf("Detected IBM version\n");


	const char *dir = "/gpfs/alpine/bif115/scratch/hjmccoy/";

	printf("Writing files to %s\n", dir);

	#else 

	const char *dir = "./";

	#endif
	
	const char *insert_op = "-insert.txt\0";
	const char *exit_lookup_op = "-exists-lookup.txt\0";
	const char *false_lookup_op = "-false-lookup.txt\0";
	const char *delete_op = "-deletions.txt\0";
	char filename_insert[256];
	char filename_exit_lookup[256];
	char filename_false_lookup[256];
	char filename_delete[256];

	bool deletions = false;

	/* Argument parsing */
	int opt;
	char *term;

	while((opt = getopt(argc, argv, "o:n:r:p:b:m:d:a:f:i:v:s:e")) != -1) {
		switch(opt) {


			case 'v':
				verbose = true;
				break;
			case 'f':
				fp_queries = strtol(optarg, &term, 10);
				break;
			case 'n':
				nbits = strtol(optarg, &term, 10);

				if (!bufset){
					buf_bits = nbits-4;
					buf_size = (1ULL << buf_bits);
				}

				if (*term) {
					fprintf(stderr, "Argument to -n must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				nslots = (1ULL << nbits);
				nvals = 950*nslots/1000;
				//buf_size = nbits - log2(npoints);
				break;
			case 'r':
				nruns = strtol(optarg, &term, 10);
				if (*term) {
					fprintf(stderr, "Argument to -r must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				break;
			case 'p':
				npoints = strtol(optarg, &term, 10);
				if (*term) {
					fprintf(stderr, "Argument to -p must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				break;
			case 'b':
				buf_bits = strtol(optarg, &term, 10);
				bufset=true;
				if (*term) {
					fprintf(stderr, "Argument to -n must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				buf_size = (1ULL << buf_bits);
				break;
			case 'd':
				datastruct = optarg;
				break;
			case 'o':
				outputfile = optarg;
				break;
			case 'e':
				deletions = true;
				break;

			default:
				fprintf(stderr, "Unknown option\n");
				usage(argv[0]);
				exit(1);
				break;
		}
	}




	if ((strcmp(datastruct, "bulk") == 0) || (strcmp(datastruct, "gqf") == 0)) {
		filter_ds = gqf;
	} else if (strcmp(datastruct, "bloom") == 0) {
		filter_ds = bloom;
	} else if (strcmp(datastruct, "point") == 0) {
		filter_ds = point;
	} else if (strcmp(datastruct, "rsqf") == 0) {
		filter_ds = rsqf;
	} else if (strcmp(datastruct, "sqf") == 0) {
		filter_ds = sqf;
	} else if (strcmp(datastruct, "coop") == 0) {
		filter_ds = coop_gqf;
	} else if (strcmp(datastruct, "cpoint") == 0) {
		filter_ds = coop_point;
	} else {
		fprintf(stderr, "Unknown filter.\n");
		usage(argv[0]);
		exit(1);
	}
	

	snprintf(filename_insert, strlen(dir) + strlen(outputfile) + strlen(insert_op) + strlen(datastruct) + 1, "%s%s%s", dir, outputfile, insert_op);
	snprintf(filename_exit_lookup, strlen(dir) + strlen(outputfile) + strlen(exit_lookup_op) + 1, "%s%s%s", dir, outputfile, exit_lookup_op);

	snprintf(filename_false_lookup, strlen(dir) + strlen(outputfile) + strlen(false_lookup_op) + 1, "%s%s%s", dir, outputfile, false_lookup_op);

	snprintf(filename_delete, strlen(dir) + strlen(outputfile) + strlen(delete_op) + 1, "%s%s%s", dir, outputfile, delete_op);


	FILE *fp_insert = fopen(filename_insert, "w");
	FILE *fp_exit_lookup = fopen(filename_exit_lookup, "w");
	FILE *fp_false_lookup = fopen(filename_false_lookup, "w");
	FILE *fp_delete = fopen(filename_delete, "w");

	if (fp_insert == NULL) {
		printf("Can't open the data file %s\n", filename_insert);
		exit(1);
	}

	if (fp_exit_lookup == NULL ) {
	    printf("Can't open the data file %s\n", filename_exit_lookup);
		exit(1);
	}

	if (fp_false_lookup == NULL) {
		printf("Can't open the data file %s\n", filename_false_lookup);
		exit(1);
	}


	for (run = 0; run < nruns; run++) {
		fps = 0;
		filter_ds.init(nbits, nbits+8, buf_size);
		

		//run setup here
		// vals_gen_state = vals_gen->init(nvals, filter_ds.range(), param);

		// if (strcmp(randmode, "zipfian_pregen") == 0) {
		// 	for (exp =0; exp < 2*npoints; exp+=2){


		// 	i = (exp/2)*(nvals/npoints);
		// 	j = ((exp/2) + 1)*(nvals/npoints);
		// 	printf("Round: %d\n", exp/2);

		// 	for (;i < j; i += 1<<16) {
		// 		int nitems = j - i < 1<<16 ? j - i : 1<<16;
		// 		__uint128_t vals[1<<16];
		// 		int m;
		// 		assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);


		// 		}

		// 	}	
		// }


		int rand_type = 0;





		//init curand here
		//setup for curand here
		//three generators - two clones for get/find and one random for fp testing
		curand_generator curand_put{};
		curand_put.init(run, rand_type, buf_size);
		curand_generator curand_get{};
		curand_get.init(run, rand_type, buf_size);
		curand_generator curand_false{};


		curand_false.init((run+1)*2702173, rand_type, buf_size);
		//curand_false.init(run, rand_type, buf_size);



		curand_put.setup_host_backing(1ULL << nbits);
		curand_get.setup_host_backing(1ULL << nbits);
		curand_false.setup_host_backing(1ULL << nbits);
		
		cudaDeviceSynchronize();

		sleep(1);




		for (exp = 0; exp < 2*npoints; exp += 2) {
			i = (exp/2)*(nvals/npoints);
			j = ((exp/2) + 1)*(nvals/npoints);
			//printf("Round: %d\n", exp/2);


			insert_times[exp+1] = std::chrono::duration<int64_t>::zero();;
			std::chrono::time_point<std::chrono::high_resolution_clock> insert_start;
			std::chrono::time_point<std::chrono::high_resolution_clock> insert_end;

			tv_insert[exp][run] = std::chrono::high_resolution_clock::now();

			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				uint64_t * vals;
				//int m;
				//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

				//prep vals for filter
				//cudaProfilerStart();

				//get timing
				insert_start = std::chrono::high_resolution_clock::now();

				curand_put.gen_next_batch(nitems);
				vals = curand_put.yield_backing();
				//cudaDeviceSynchronize();
				insert_end = std::chrono::high_resolution_clock::now();
				
				insert_times[exp+1] += insert_end-insert_start;

			
					
					
				filter_ds.bulk_insert(vals, nitems);
					//cudaProfilerStop();

				
			}

			cudaDeviceSynchronize();


			tv_insert[exp+1][run] = std::chrono::high_resolution_clock::now();

			//don't need this
			//curand_test.reset_to_defualt();

			exit_times[exp+1] =  std::chrono::duration<int64_t>::zero();
			std::chrono::time_point<std::chrono::high_resolution_clock> exit_start;
			std::chrono::time_point<std::chrono::high_resolution_clock> exit_end;

			

			i = (exp/2)*(nvals/npoints);
			tv_exit_lookup[exp][run]= std::chrono::high_resolution_clock::now();
			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				
				//assert(vals_gen->gen(old_vals_gen_state, nitems, vals) == nitems);
			
		
				//int m;
				//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);
				uint64_t * insert_vals;
				//prep vals for filter

				exit_start = std::chrono::high_resolution_clock::now();
				curand_get.gen_next_batch(nitems);
				insert_vals = curand_get.yield_backing();
				exit_end = std::chrono::high_resolution_clock::now();
				exit_times[exp+1] += exit_end-exit_start;

				uint64_t result = filter_ds.bulk_lookup(insert_vals, nitems);
				//uint64_t result = 0;
				if (result != 0){

				printf("Failed to find %llu items\n", result);
				abort();

				}

			}

			cudaDeviceSynchronize();
			tv_exit_lookup[exp+1][run] = std::chrono::high_resolution_clock::now();

			

			//this looks right
			false_times[exp+1] = std::chrono::duration<int64_t>::zero();;
			std::chrono::time_point<std::chrono::high_resolution_clock> false_start;
			std::chrono::time_point<std::chrono::high_resolution_clock> false_end;

			//curand_test.destroy();
			//curand_generator othervals_curand{};
			//othervals_curand.init_curand(5, 0, buf_size);

			i = (exp/2)*(nvals/npoints);
			tv_false_lookup[exp][run] = std::chrono::high_resolution_clock::now();
			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				uint64_t * othervals;

				false_start = std::chrono::high_resolution_clock::now();
				curand_false.gen_next_batch(nitems);
				othervals = curand_false.yield_backing();
				false_end = std::chrono::high_resolution_clock::now();
				false_times[exp+1] += false_end-false_start;

		
				if (!fp_queries){


					
				//fps += nitems-filter_ds.bulk_lookup(othervals, nitems);
				filter_ds.bulk_lookup(othervals, nitems);

				} else {

					fps += nitems-filter_ds.bulk_fp_lookup(othervals, nitems);

				}
					
				
			}

			cudaDeviceSynchronize();
			tv_false_lookup[exp+1][run] = std::chrono::high_resolution_clock::now();
			
			
		}


		//and destroy

		//all inserts done, reset main counter




		curand_put.destroy();
		curand_get.destroy();
		curand_false.destroy();

		
		
 
		//
		// for (exp = 0; exp < 2*npoints; exp += 2) {
		// 	i = (exp/2)*(nvals/npoints);
		// 	j = ((exp/2) + 1)*(nvals/npoints);
		// 	//printf("Round: %d\n", exp/2);


		// 	for (;i < j; i += buf_size) {
		// 		int nitems = j - i < buf_size ? j - i : buf_size;
		// 		uint64_t * vals;
		// 		//int m;
		// 		//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

		// 		//prep vals for filter
		// 		//cudaProfilerStart();
		// 		get_end.gen_next_batch(nitems);
		// 		vals = get_end.yield_backing();

		// 		uint64_t result = filter_ds.bulk_lookup(vals, nitems);
		// 		if (result != 0){

		// 			printf("Failed to find %llu items\n", result);
		// 			abort();

		// 		}


		// 	}	

		// }

		curand_generator get_end{};
		get_end.init(run, rand_type, buf_size);

		get_end.setup_host_backing(1ULL << nbits);
		
		for (exp = 0; exp < 2*npoints; exp += 2) {
			i = (exp/2)*(nvals/npoints);
			j = ((exp/2) + 1)*(nvals/npoints);
			//printf("Round: %d\n", exp/2);


			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				uint64_t * vals;
				//int m;
				//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

				//prep vals for filter
				//cudaProfilerStart();
				get_end.gen_next_batch(nitems);
				vals = get_end.yield_backing();

				//this should always trigger fp - need to assert that items are correct
				uint64_t result = filter_ds.bulk_fp_lookup(vals, nitems);
				if (result != 0){

					printf("Failed to find %llu items\n", result);
					abort();

				}


			}	

		}
		
		get_end.destroy();

		// for (exp = 0; exp < 2*npoints; exp += 2) {
		// 	i = (exp/2)*(nvals/npoints);
		// 	j = ((exp/2) + 1)*(nvals/npoints);
		// 	//printf("Round: %d\n", exp/2);


		// 	delete_times[exp+1] = std::chrono::duration<int64_t>::zero();


		// 	std::chrono::time_point<std::chrono::high_resolution_clock> delete_start;
		// 	std::chrono::time_point<std::chrono::high_resolution_clock> delete_end;



		// 	auto outer = std::chrono::high_resolution_clock::now();
		// 	tv_delete[exp][run] = std::chrono::high_resolution_clock::now();

		// 	for (;i < j; i += buf_size) {
		// 		int nitems = j - i < buf_size ? j - i : buf_size;
		// 		uint64_t * vals;
		// 		//int m;
		// 		//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

		// 		//prep vals for filter
		// 		//cudaProfilerStart();

		// 		//get timing
		// 		delete_start = std::chrono::high_resolution_clock::now();

		// 		get_end.gen_next_batch(nitems);
		// 		vals = get_end.yield_backing();
		// 		//cudaDeviceSynchronize();
		// 		delete_end = std::chrono::high_resolution_clock::now();

		// 		//printf("Time taken in setup: %llu\n", (delete_end - delete_start).count());
				
		// 		delete_times[exp+1] += delete_end-delete_start;

			
					
					
		// 		filter_ds.bulk_delete(vals, nitems);
		// 			//cudaProfilerStop();

				
		// 	}

		// 	cudaDeviceSynchronize();


		// 	auto outer_end = std::chrono::high_resolution_clock::now();
		// 	delete_times[exp+1] = (outer_end - outer) - delete_times[exp+1];

		// 	//printf("Time taken: %llu\n", std::chrono::duration_cast<std::chrono::seconds>(delete_times[exp+1]).count());

		// }



		//get_end.destroy();
		

		if (deletions){

			curand_generator get_deletes{};
			get_deletes.init(run, rand_type, buf_size);


			for (exp = 0; exp < 2*npoints; exp += 2) {
				i = (exp/2)*(nvals/npoints);
				j = ((exp/2) + 1)*(nvals/npoints);
				//printf("Round: %d\n", exp/2);


				delete_times[exp+1] = std::chrono::duration<int64_t>::zero();
				std::chrono::time_point<std::chrono::high_resolution_clock> insert_start;
				std::chrono::time_point<std::chrono::high_resolution_clock> insert_end;

				tv_delete[exp][run] = std::chrono::high_resolution_clock::now();

				for (;i < j; i += buf_size) {
					int nitems = j - i < buf_size ? j - i : buf_size;
					uint64_t * vals;
					//int m;
					//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

					//prep vals for filter
					//cudaProfilerStart();

					//get timing
					insert_start = std::chrono::high_resolution_clock::now();

					get_deletes.gen_next_batch(nitems);
					vals = get_deletes.yield_backing();
					//cudaDeviceSynchronize();
					insert_end = std::chrono::high_resolution_clock::now();
					
					delete_times[exp+1] += insert_end-insert_start;

				
						
						
					filter_ds.bulk_delete(vals, nitems);
						//cudaProfilerStop();


					
				}

				cudaDeviceSynchronize();


				tv_delete[exp+1][run] = std::chrono::high_resolution_clock::now();



			}

			get_deletes.destroy();

			

			if (verbose) printf("Deletion Tests Complete\n");
		}


		cudaDeviceSynchronize();

		if (deletions){


		curand_generator get_final{};
		get_final.init(run, rand_type, buf_size);
		
		for (exp = 0; exp < 2*npoints; exp += 2) {
			i = (exp/2)*(nvals/npoints);
			j = ((exp/2) + 1)*(nvals/npoints);
			//printf("Round: %d\n", exp/2);


			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				uint64_t * vals;
				//int m;
				//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

				//prep vals for filter
				//cudaProfilerStart();
				get_final.gen_next_batch(nitems);
				vals = get_final.yield_backing();

				uint64_t result;

		

				result = filter_ds.bulk_fp_lookup(vals, nitems);
				

				
				if (result != nitems){

					printf("Failed to not find %llu/%llu items after deletion\n", nitems-result, nitems);
					//abort();

				}


			}	

		}
		
		get_final.destroy();

		cudaDeviceSynchronize();
		}
		

		filter_ds.destroy();
		cudaDeviceSynchronize();


	}





	if (verbose) printf("Writing results to file: %s\n",  filename_insert);


	fprintf(fp_insert, "x_0");
	for (run = 0; run < nruns; run++) {
		fprintf(fp_insert, "    y_%d", run);
	}
	fprintf(fp_insert, "\n");
	for (exp = 0; exp < 2*npoints; exp += 2) {
		fprintf(fp_insert, "%d", ((exp/2)*(100/npoints)));
		for (run = 0; run < nruns; run++) {
			fprintf(fp_insert, " %f",
							0.001 * (nvals/npoints)/ ((tv_insert[exp+1][run] - tv_insert[exp][run])-insert_times[exp+1]).count()*1000000);
		}
		fprintf(fp_insert, "\n");
	}
	if (verbose) printf("Insert Performance written\n");

	if (verbose) printf("Writing results to file: %s\n", filename_exit_lookup);
	fprintf(fp_exit_lookup, "x_0");
	for (run = 0; run < nruns; run++) {
		fprintf(fp_exit_lookup, "    y_%d", run);
	}
	fprintf(fp_exit_lookup, "\n");
	for (exp = 0; exp < 2*npoints; exp += 2) {
		fprintf(fp_exit_lookup, "%d", ((exp/2)*(100/npoints)));
		for (run = 0; run < nruns; run++) {
			fprintf(fp_exit_lookup, " %f",
							0.001 * (nvals/npoints)/((tv_exit_lookup[exp+1][run]- tv_exit_lookup[exp][run])-exit_times[exp+1]).count()*1000000);
		}
		fprintf(fp_exit_lookup, "\n");
	}
	if (verbose) printf("Existing Lookup Performance written\n");

	if (verbose) printf("Writing results to file: %s\n", filename_false_lookup);
	fprintf(fp_false_lookup, "x_0");
	for (run = 0; run < nruns; run++) {
		fprintf(fp_false_lookup, "    y_%d", run);
	}
	fprintf(fp_false_lookup, "\n");
	for (exp = 0; exp < 2*npoints; exp += 2) {
		fprintf(fp_false_lookup, "%d", ((exp/2)*(100/npoints)));
		for (run = 0; run < nruns; run++) {
			fprintf(fp_false_lookup, " %f",
							0.001 * (nvals/npoints)/((tv_false_lookup[exp+1][run]- tv_false_lookup[exp][run])-false_times[exp+1]).count()*1000000);
		}
		fprintf(fp_false_lookup, "\n");
	}

	if (verbose) printf("False Lookup Performance written\n");

	if (fp_queries) printf("FP rate: %f (%lu/%lu)\n", 1.0 * fps / nvals, fps, nvals);


	if (deletions){



	if (verbose) printf("Writing results to file: %s\n", filename_delete);
	fprintf(fp_delete, "x_0");
	for (run = 0; run < nruns; run++) {
		fprintf(fp_delete, "    y_%d", run);
	}
	fprintf(fp_delete, "\n");
	for (exp = 0; exp < 2*npoints; exp += 2) {
		fprintf(fp_delete, "%d", ((exp/2)*(100/npoints)));

		for (run = 0; run < nruns; run++) {
		fprintf(fp_delete, " %f",
						0.001 * (nvals/npoints)/((tv_delete[exp+1][run]- tv_delete[exp][run])-delete_times[exp+1]).count()*1000000);
		}
		// for (run = 0; run < nruns; run++) {

		// 	//printf("Delete: %f \n", ((tv_delete[exp+1][run]- tv_delete[exp][run])).count());
		// 	//this is all bugged out but the math is the same
		// 	//for some reason delete timers kept getting reset to 0
		// 	fprintf(fp_delete, " %f",
		// 					0.001 * (nvals/npoints)/(delete_times[exp+1]).count());
		// }
		fprintf(fp_delete, "\n");
	}
	if (verbose) printf("Delete Performance written\n");

	//printf("FP rate: %f (%lu/%lu)\n", 1.0 * fps / nvals, fps, nvals);



	}

	fclose(fp_insert);
	fclose(fp_exit_lookup);
	fclose(fp_false_lookup);

	return 0;
}

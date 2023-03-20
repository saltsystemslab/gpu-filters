/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This file is for isolating a bug in the TCF delete marking
 *          - insert a few items, measure fill, delete, and then recalculate.
 *          
 *
 * ============================================================================
 */



#include "include/gqf_wrapper.cuh"
#include "include/point_wrapper.cuh"


#include <stdio.h>
#include <iostream>
#include <chrono>
#include <openssl/rand.h>

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


double elapsed(std::chrono::high_resolution_clock::time_point t1, std::chrono::high_resolution_clock::time_point t2) {
   return (std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1)).count();
}


// using insert_type = poggers::insert_schemes::single_slot_insert<uint64_t, uint64_t, 8, 8, poggers::representations::key_val_pair, 5, poggers::hashers::murmurHasher, poggers::probing_schemes::doubleHasher>;

// using table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 4, 4, poggers::insert_schemes::bucket_insert, 200, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
//      // poggers::representations::key_val_pair, 8>

//      //using forst_tier_table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
    
// using second_tier_table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, table_type>;

// using inner_table = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

// using small_double_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, inner_table>;

// using p2_table = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 8, 16, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   
// using tier_one_iceberg = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 1, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher>;

// using tier_two_icerberg = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::powerOfTwoHasher, poggers::hashers::murmurHasher>;

// using tier_three_iceberg = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 10, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


// using tier_two_icerberg_joined = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::powerOfTwoHasher, poggers::hashers::murmurHasher, true, tier_three_iceberg>;

// using iceberg_table = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 64, poggers::insert_schemes::bucket_insert, 1, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher, true, tier_two_icerberg_joined>;


// using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
// using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tiny_static_table_4>;

// using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
// using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::XORPowerOfTwoHasher, poggers::hashers::murmurHasher, true, tiny_static_table_4>;



// shortened_key_val_wrapper


//using double_buckets = poggers::tables::bucketed_table<uint64_t, uint64_t, poggers::representations::struct_of_arrays, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals = (T *) malloc(nitems * sizeof(T));


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   return vals;
}


template <typename Key, typename Val>
__global__ void speed_insert_kernel(QF * gqf, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= nvals) return;

   if (point_insert(gqf, keys[tid], vals[tid], 0) != QF_ITEM_INSERTED){

      atomicAdd((unsigned long long *) misses, 1ULL);

   }
   //assert(filter->insert(tile, keys[tid], vals[tid]));


}


template <typename Key, typename Val>
__global__ void speed_delete_kernel(QF * gqf, Key * keys, Val * vals, uint64_t nvals, uint64_t * del_misses, uint64_t * del_failures){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= nvals) return;
   
   if (point_delete(gqf, keys[tid], vals[tid], 0) == -1){

      atomicAdd((unsigned long long *) del_misses, 1ULL);

   }
   //assert(filter->insert(tile, keys[tid], vals[tid]));


}


template <typename Key, typename Val>
__global__ void speed_query_kernel(QF * gqf, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){



   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= nvals) return;



   uint64_t local_val;

   if (point_query(gqf, keys[tid], local_val, 0) == 0){

      atomicAdd((unsigned long long int *) query_misses, 1ULL);

   } else {

      if (point_query_count(gqf, keys[tid], vals[tid], 0) == 0){

         //failures are fp - so key was found but with different value.

         atomicAdd((unsigned long long int *)query_failures, 1ULL);

      }

   }
   //assert(filter->insert(tile, keys[tid], vals[tid]));


}








template <typename Key, typename Val>
__host__ void point_sawtooth_test(int num_bits, int num_partitions, int num_rounds){

   //to start I'm not considering cases where we delete beyond 100% of input.
   //assert(num_rounds < num_partitions);

   uint64_t nitems = (1ULL << num_bits)*.85;

   //Initializer->full_reset();


   uint64_t items_per_partition = nitems/num_partitions;


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

   Val * dev_vals;

   cudaMalloc((void **)&dev_vals, sizeof(Val)*items_per_partition);

   //static seed for testing
   QF * gqf;

   qf_malloc_device(&gqf, num_bits, 8, 0, false);

   Val * host_vals = generate_data<Val>(items_per_partition);

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

      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      speed_insert_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses);
   

      //speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition),test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, dev_vals, items_per_partition, misses);
   

      cudaDeviceSynchronize();

   }


   for (int i = 0; i < num_rounds; i++){


      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      speed_delete_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+1, misses+2);
   
      cudaDeviceSynchronize();

      //false negative query
      for (int j = i+1; j < num_partitions; j++){

         cudaMemcpy(dev_keys, inserted_keys[j], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();

         speed_query_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+3, misses+4);
   
         cudaDeviceSynchronize();
      }


      for (int k = 0; k<i; k++){


         cudaMemcpy(dev_keys, new_keys[k], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();

         speed_query_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+3, misses+4);
   

      }

      //and finally re-add

      cudaDeviceSynchronize();

      cudaMemcpy(dev_keys, new_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      speed_insert_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+5);
   
      cudaDeviceSynchronize();
   


   }

   printf("Insert fails: %llu, delete misses: %llu, delete false matching: %llu, False negatives: %llu, false positives: %llu, re-add misses %llu\n", misses[0], misses[1], misses[2], misses[3], misses[4], misses[5]);

   return;



   // //and delete half


 
   // double insert_throughput = 1.0*nitems/elapsed(insert_start, insert_end);

   // double query_throughput = .5*nitems/elapsed(query_start, query_end);

   // double delete_throughput = .5*nitems/elapsed(delete_start, delete_end);

   // double readd_throughput = .5*nitems/elapsed(readd_start, readd_end);

   // printf("Insert throughput: %f, delete throughput: %f, query throughput: %f, readd throughput: %f\n", insert_throughput, delete_throughput, query_throughput, readd_throughput);


   //printf("Fill before: %llu, fill_after: %llu, %f ... Final fill: %llu\n", fill_before, fill_after, 1.0*fill_after/fill_before, final_fill);
   cudaDeviceSynchronize();

   cudaFree(misses);

   cudaFree(dev_keys);

   cudaFree(dev_vals);

   qf_destroy_device(gqf);

}



template <typename Key, typename Val>
__host__ void bulk_sawtooth_test(int num_bits, int num_partitions, int num_rounds){

   //to start I'm not considering cases where we delete beyond 100% of input.
   //assert(num_rounds < num_partitions);

   uint64_t nitems = (1ULL << num_bits)*.85;

   //Initializer->full_reset();


   uint64_t items_per_partition = nitems/num_partitions;


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

   Val * dev_vals;

   cudaMalloc((void **)&dev_vals, sizeof(Val)*items_per_partition);

   //static seed for testing
   QF * gqf;

   qf_malloc_device(&gqf, num_bits, 8, 0, true);

   Val * host_vals = generate_data<Val>(items_per_partition);

   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*6);
   cudaDeviceSynchronize();


   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;
   misses[5] = 0;


   printf("Starting bulk sawtooth test for %llu items, %llu items per %d partitions\n", nitems, items_per_partition, num_partitions);

   cudaDeviceSynchronize();


   for (int i = 0; i < num_partitions; i++){

      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();


      bulk_insert_values(gqf, items_per_partition, dev_keys, dev_vals, 0);   

      //speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition),test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, dev_vals, items_per_partition, misses);
   
      cudaDeviceSynchronize();

   }


   for (int i = 0; i < num_rounds; i++){


      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      bulk_delete_values(gqf, items_per_partition, dev_keys, dev_vals, 0);

      //speed_delete_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+1, misses+2);
   
      cudaDeviceSynchronize();

      printf("Done with delete round %d\n", i);

      //false negative query
      for (int j = i+1; j < num_partitions; j++){

         cudaMemcpy(dev_keys, inserted_keys[j], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);


         cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();


         misses[3] += bulk_get_exact_misses_wrapper(gqf, dev_keys, dev_vals, items_per_partition);
         //speed_query_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+3, misses+4);
   
         cudaDeviceSynchronize();
      }


      for (int k = 0; k<i; k++){


         cudaMemcpy(dev_keys, new_keys[k], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);


         cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();


         misses[3] += bulk_get_exact_misses_wrapper(gqf, dev_keys, dev_vals, items_per_partition);

         //speed_query_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+3, misses+4);
   

      }

      printf("Done with queries round %d\n", i);

      //and finally re-add

      cudaDeviceSynchronize();

      cudaMemcpy(dev_keys, new_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      bulk_insert_values(gqf, items_per_partition, dev_keys, dev_vals, 0);
      //speed_insert_kernel<Key, Val><<<(items_per_partition-1)/512+1, 512>>>(gqf, dev_keys, dev_vals, items_per_partition, misses+5);
   
      cudaDeviceSynchronize();


      printf("Done with re-insert round %d\n", i);
   


   }

   printf("Insert fails: %llu, delete misses: %llu, delete false matching: %llu, False negatives: %llu, false positives: %llu, re-add misses %llu\n", misses[0], misses[1], misses[2], misses[3], misses[4], misses[5]);

   return;

   cudaDeviceSynchronize();

   cudaFree(misses);

   cudaFree(dev_keys);

   cudaFree(dev_vals);

   qf_destroy_device(gqf);

}




int main(int argc, char** argv) {



   if (argc < 4) {
      fprintf(stderr, "Please specify the log of the number of items to test with.\n");
      exit(1);

   }

   int nbits = atoi(argv[1]);

   int num_partitions = atoi(argv[2]);

   int num_rounds = atoi(argv[3]);


   point_sawtooth_test<uint64_t, uint8_t>(nbits, num_partitions, num_rounds);

   bulk_sawtooth_test<uint64_t, uint64_t>(nbits, num_partitions, num_rounds);

   cudaDeviceSynchronize();

   return 0;

}

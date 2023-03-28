/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This tests the performance of the TCF on the sawtooth test
 *          by filling to a high load factor than incrementally deleting/reinserting.
 *          
 *
 * ============================================================================
 */




//#include "include/templated_quad_table.cuh"
#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/linear_probing.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/probing_schemes/power_of_two.cuh>
#include <poggers/insert_schemes/single_slot_insert.cuh>
#include <poggers/insert_schemes/bucket_insert.cuh>
#include <poggers/insert_schemes/power_of_n.cuh>
#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/shortened_key_val_pair.cuh>
#include <poggers/sizing/default_sizing.cuh>
#include <poggers/tables/base_table.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut.cuh>

#include <poggers/representations/soa.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut_buckets.cuh>

#include <poggers/tables/bucketed_table.cuh>

#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/probing_schemes/power_of_two.cuh>

// new container for 2-byte key val pairs
#include <poggers/representations/grouped_key_val_pair.cuh>

#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/dynamic_container.cuh>

#include <poggers/sizing/default_sizing.cuh>

#include <poggers/insert_schemes/power_of_n_shortcut.cuh>

#include <poggers/insert_schemes/power_of_n_shortcut_buckets.cuh>

#include <poggers/representations/packed_bucket.cuh>

#include <poggers/insert_schemes/linear_insert_buckets.cuh>

#include <poggers/tables/bucketed_table.cuh>

#include <poggers/representations/grouped_storage_sub_bits.cuh>

#include <poggers/probing_schemes/xor_power_of_two.cuh>


#include <poggers/data_structs/tcf.cuh>


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

using del_TCF = poggers::data_structs::tcf_wrapper<uint64_t, uint16_t, 16, 16, 4, 16>::tcf;



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


template <typename Filter, typename Key, typename Val>
__global__ void print_tid_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals){


      auto tile = filter->get_my_tile();

      uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

      if (tid >= nvals) return;


      if (tile.thread_rank() == 0) printf("%llu: %d, %d\n", tid, keys[tid], vals[tid]);

}

template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->insert(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   } else{

      Val test_val = 0;
      test_val+=0;
      assert(filter->query(tile, keys[tid], test_val));
   }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_with_delete_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->insert_with_delete(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   } else{

      Val test_val = 0;
      test_val+=0;
      assert(filter->query(tile, keys[tid], test_val));
   }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel_one_thread(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid2 = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid2 > 0) return;


   for (uint64_t tid=0; tid < nvals; tid++){

      if (tid % 10 == 0 && tile.thread_rank() == 0) printf("%llu\n", tid);

      if (!filter->insert(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){

      atomicAdd((unsigned long long int *) misses, 1ULL);
    } else {

      filter->remove(tile, keys[tid]);
    }
    //else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));
   // }


   }


   //assert(filter->insert(tile, keys[tid], vals[tid]));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_delete_kernel(Filter * filter, Key * keys, uint64_t nvals, uint64_t * del_misses, uint64_t * del_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   if (!filter->remove(tile,keys[tid]) ){

      Val val = 0;
      val+=0;

      filter->query(tile, keys[tid], val);
      filter->remove(tile, keys[tid]);

      filter->query(tile, keys[tid], val);

      if ( tile.thread_rank() == 0) atomicAdd((unsigned long long int *) del_misses, 1ULL);

   } else {

      Val val =0;
      //thank you compiler very cool
      val +=0 ;
      if (filter->query(tile,keys[tid], val) && tile.thread_rank() == 0 ){

         //this is not necessarily a failure to delete, but is indicative of a false-positive match
         atomicAdd((unsigned long long int *) del_failures, 1ULL);

      }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_delete_single_thread(Filter * filter, Key * keys, uint64_t nvals, uint64_t * del_misses, uint64_t * del_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid != 0) return;


   for (uint64_t i=0; i < nvals; i++){

      if (!filter->remove(tile,keys[i]) ){

      Val val;
      val+=0;

      filter->query(tile, keys[i], val);
      filter->remove(tile, keys[i]);

      filter->query(tile, keys[i], val);

      if ( tile.thread_rank() == 0) atomicAdd((unsigned long long int *) del_misses, 1ULL);

   } else {

      Val val;
      //thank you compiler very cool
      val +=0 ;
      if (filter->query(tile,keys[i], val) && tile.thread_rank() == 0 ){

         atomicAdd((unsigned long long int *) del_failures, 1ULL);

      }

   }


   }
   
   //assert(filter->query(tile, keys[tid], val));


}



template <typename Filter, typename Key, typename Val>
__global__ void speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val val = 0;
   val += 0;

   if (!filter->query(tile,keys[tid], val) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) query_misses, 1ULL);
   } else {

      if (val != vals[tid] && tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *) query_failures, 1ULL);
      }

   }
   //assert(filter->query(tile, keys[tid], val));


}

template <typename Filter, typename Key, typename Val, typename Sizing_Type>
__host__ void test_speed(Sizing_Type * Initializer){

   uint64_t nitems = Initializer->total()*.9;

   Key * host_keys = generate_data<Key>(nitems);
   Val * host_vals = generate_data<Val>(nitems);

   Key * dev_keys;

   Val * dev_vals;

   cudaMalloc((void **)& dev_keys, nitems*sizeof(Key));
   cudaMalloc((void **)& dev_vals, nitems*sizeof(Val));

   cudaMemcpy(dev_keys, host_keys, nitems*sizeof(Key), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, nitems*sizeof(Val), cudaMemcpyHostToDevice);


   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*3);
   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;

   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);

   cudaDeviceSynchronize();

   auto insert_start = std::chrono::high_resolution_clock::now();

   //add function for configure parameters - should be called by ht and return dim3
   speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems, misses);
   cudaDeviceSynchronize();
   auto insert_end = std::chrono::high_resolution_clock::now();


   cudaMemcpy(dev_keys, host_keys, nitems*sizeof(Key), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, nitems*sizeof(Val), cudaMemcpyHostToDevice);


   cudaDeviceSynchronize();

   auto query_start = std::chrono::high_resolution_clock::now();

   speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems, &misses[1], &misses[2]);
   cudaDeviceSynchronize();
   auto query_end = std::chrono::high_resolution_clock::now();




   std::chrono::duration<double> insert_diff = insert_end-insert_start;
   std::chrono::duration<double> query_diff = query_end-query_start;


   cudaDeviceSynchronize();
   std::cout << "Inserted " << nitems << " in " << insert_diff.count() << " seconds\n";
   std::cout << "Queried " << nitems << " in " << query_diff.count() << " seconds\n";

   printf("Inserts/Queries: %f / %f\n", 1.0*nitems/insert_diff.count(), 1.0*nitems/query_diff.count());
   printf("%llu / %llu / %llu\n", misses[0], misses[1], misses[2]);

   cudaDeviceSynchronize();

   cudaFree(misses);

   cudaDeviceSynchronize();

   cudaFree(dev_keys);
   cudaFree(dev_vals);

   Filter::free_on_device(test_filter);

   free(host_keys);
   free(host_vals);

}


template <typename Filter, typename Key, typename Val>
__global__ void delete_insert_kernel(Filter * filter){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0)return;

   uint64_t key = 1;

   uint8_t  val = 0;

   auto tile = filter->get_my_tile();

   filter->insert(tile, key, val);

}

template <typename Filter, typename Key, typename Val>
__global__ void delete_delete_kernel(Filter * filter){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0)return;

   uint64_t key = 1;

   uint8_t  val = 0;

   auto tile = filter->get_my_tile();

   filter->remove(tile, key);

}

template <typename Filter, typename Key, typename Val, typename Sizing_Type>
__host__ void test_del_batched(Sizing_Type * Initializer){


   std::cout << "Starting test\n";

   uint64_t nitems = Initializer->total();




   cudaDeviceSynchronize();


   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();


   printf("Fill before %llu\n", test_filter->get_fill());


   cudaDeviceSynchronize();

   delete_insert_kernel<Filter, Key, Val><<<1,1>>>(test_filter);

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);

   cudaDeviceSynchronize();

   printf("Fill after insert %llu\n", test_filter->get_fill());


   cudaDeviceSynchronize();

   delete_delete_kernel<Filter, Key, Val><<<1,1>>>(test_filter);

   cudaDeviceSynchronize();

   printf("Fill after delete %llu\n", test_filter->get_fill());

   Filter::free_on_device(test_filter);


   //free pieces

   //time to output

}


template <typename Filter, typename Key, typename Val, typename Sizing_Type>
__host__ void sawtooth_test(Sizing_Type * Initializer, int num_partitions, int num_rounds){

   //to start I'm not considering cases where we delete beyond 100% of input.
   //assert(num_rounds < num_partitions);

   uint64_t nitems = Initializer->total()*.85;

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
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

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

      printf("Starting setup round %d, fill %llu\n",i, test_filter->get_fill());

      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition),test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, dev_vals, items_per_partition, misses);
   

      //speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition),test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, dev_vals, items_per_partition, misses);
   

      cudaDeviceSynchronize();

   }


   for (int i = 0; i < num_rounds; i++){

      printf("Starting sawtooth round %d, fill %llu\n",i, test_filter->get_fill());

      cudaMemcpy(dev_keys, inserted_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

      cudaMemcpy(dev_vals, host_vals, sizeof(Val)*items_per_partition, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      speed_delete_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition), test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, items_per_partition, misses+1, misses+2);

      cudaDeviceSynchronize();


      //false negative query
      for (int j = i+1; j < num_partitions; j++){

         cudaMemcpy(dev_keys, inserted_keys[j], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();

         speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition), test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, dev_vals, items_per_partition, misses+3, misses+4);

         cudaDeviceSynchronize();
      }


      for (int k = 0; k<i; k++){


         cudaMemcpy(dev_keys, new_keys[k], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);

         cudaDeviceSynchronize();

         speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition), test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, dev_vals, items_per_partition, misses+3, misses+4);


      }

      //and finally re-add

      cudaDeviceSynchronize();

      cudaMemcpy(dev_keys, new_keys[i], sizeof(Key)*items_per_partition, cudaMemcpyHostToDevice);


      speed_insert_with_delete_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(items_per_partition),test_filter->get_block_size(items_per_partition)>>>(test_filter, dev_keys, dev_vals, items_per_partition, misses+5);

      cudaDeviceSynchronize();
   


   }

   printf("Insert fails: %llu, delete misses: %llu, delete value mismatch: %llu, False negatives: %llu, false positives: %llu, re-add misses %llu\n", misses[0], misses[1], misses[2], misses[3], misses[4], misses[5]);

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

   Filter::free_on_device(test_filter);

}



// __host__ void test_p2(uint64_t nitems){

//    printf("size: %llu\n", nitems);
//    poggers::sizing::size_in_num_slots<1>half_split_20(nitems);
//    test_speed<p2_table, uint64_t, uint16_t>(&half_split_20);
// }



int main(int argc, char** argv) {

   // poggers::sizing::size_in_num_slots<1> first_size_20(1ULL << 20);
   // printf("2^20\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_20);

   // poggers::sizing::size_in_num_slots<1> first_size_22(1ULL << 22);
   // printf("2^22\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_22);

   // poggers::sizing::size_in_num_slots<1> first_size_24(1ULL << 24);
   // printf("2^24\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_24);

   // poggers::sizing::size_in_num_slots<1> first_size_26(1ULL << 26);
   // printf("2^26\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_26);

   // poggers::sizing::size_in_num_slots<1> first_size_28(1ULL << 28);
   // printf("2^28\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_28);

   if (argc < 4) {
      fprintf(stderr, "Please specify the log of the number of items to test with.\n");
      exit(1);

   }

   int nbits = atoi(argv[1]);

   int num_partitions = atoi(argv[2]);

   int num_rounds = atoi(argv[3]);


   poggers::sizing::size_in_num_slots<2> bucket_size_2 (1ULL<<nbits, (1ULL <<nbits)/100);


   sawtooth_test<del_TCF, uint64_t, uint16_t>(&bucket_size_2, num_partitions, num_rounds);


   // for (int i = 0; i< 1; i++){

     

   //    delete_tests<del_TCF_small, uint64_t, uint8_t>(&bucket_size_2);

   // }

   // poggers::sizing::variadic_size test_size_24_tcf ((1ULL << nbits), (1ULL << nbits)/100);

   // //printf("22 size: %llu\n", test_size_24.total());
   //test_speed_batched<tcf, uint64_t, uint16_t>("results/test_32", &test_size_24_tcf, 20);



   //poggers::sizing::size_in_num_slots<1> bucket_size (1ULL<<nbits);

   //test_speed_batched<double_buckets, uint64_t,uint64_t>("results/double_buckets", &bucket_size, 20);

   cudaDeviceSynchronize();

   // printf("alt table\n");

   // test_p2(6000);

   // test_p2(1ULL << 22);
   // test_p2(1ULL << 24);
   // test_p2(1ULL << 26);
   // test_p2(1ULL << 28);
   // test_p2(1ULL << 30);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_22);

   // poggers::sizing::size_in_num_slots<2>half_split_24(1ULL << 23, 1ULL << 23);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_24);

   // poggers::sizing::size_in_num_slots<2>half_split_26(1ULL << 25, 1ULL << 25);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_26);


   // printf("P2 tiny table\n");
   // poggers::sizing::size_in_num_slots<1>half_split_28(1ULL << 30);
   // test_speed<p2_table, uint64_t, uint16_t>(&half_split_28);


   //printf("Icerberg - Tier one\n");


   //poggers::sizing::size_in_num_slots<1> tier_one_iceberg_size(1ULL << 28);
   //test_speed<tier_one_iceberg, uint64_t, uint64_t>(&tier_one_iceberg_size);


   // printf("Icerberg - Tier two\n");

   // //this section is allocated 1/8th of the space as tier one
   // poggers::sizing::size_in_num_slots<1> tier_two_iceberg_size((1ULL << 28)/8);
   // test_speed<tier_two_icerberg, uint64_t, uint64_t>(&tier_two_iceberg_size);

   // printf("Icerberg - Tier three\n");

   // poggers::sizing::size_in_num_slots<1> tier_three_iceberg_size((1500));
   // test_speed<tier_three_iceberg, uint64_t, uint64_t>(&tier_three_iceberg_size);


   // printf("Icerberg - Joined\n");

   // poggers::sizing::size_in_num_slots<3> iceberg_size((1ULL << 28), (1ULL << 28)/8, 1500);
   // test_speed<iceberg_table, uint64_t, uint64_t>(&iceberg_size);

   //del_TCF test;

	return 0;

}

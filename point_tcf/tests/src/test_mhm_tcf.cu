#include "hip/hip_runtime.h"
/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This file contains speed tests for several Hash Table Types
 *          built using POGGERS. For more verbose testing please see the 
 *          benchmarks folder.
 *
 * ============================================================================
 */


#include <poggers/data_structs/tcf.cuh>

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <openssl/rand.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
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


// using backing_table = poggers::tables::bucketed_table<
//     uint64_t, uint8_t,
//     poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
//         poggers::representations::bit_grouped_container<10, 6>::representation, uint16_t>::representation>::representation,
//     1, 8, poggers::insert_schemes::linear_insert_bucket_scheme, 20, poggers::probing_schemes::doubleHasher,
//     poggers::hashers::murmurHasher>;
// using TCF = poggers::tables::bucketed_table<
//     uint64_t, uint8_t,
//     poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
//         poggers::representations::bit_grouped_container<10, 6>::representation, uint16_t>::representation>::representation,
//     1, 8, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher,
//     poggers::hashers::murmurHasher, true, backing_table>;


using TCF = poggers::data_structs::tcf_wrapper<uint64_t, uint8_t, 26, 6, 1, 16>::tcf;


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


uint64_t num_slots_per_p2(uint64_t nitems){

   //uint64_t nitems = .9*(1ULL << nbits);

   //for p=1/100, this is the correct value

   uint64_t nslots = 959*nitems/100;
   printf("using %llu slots\n", nslots);
   return nslots; 

}


template <typename T>
__host__ T * load_main_data(uint64_t nitems){


   char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/main_data-32-data.txt";

   //char main_location[] = "/pscratch/sd/h/hunterm/vqf_data/main_data-32-data.txt";

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


   //    getline(myfile, line);

   //    strncpy(vals, line.c_str(), sizeof(uint64_t)*nitems);

   //    myfile.close();
      

   // } else {

   //    abort();
   // }


   return (T *) vals;


}

template <typename T>
__host__ T * load_alt_data(uint64_t nitems){


   char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/fp_data-32-data.txt";

   //char main_location[] = "/pscratch/sd/h/hunterm/vqf_data/fp_data-32-data.txt";


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
__global__ void find_first_fill(Filter * filter, Key * keys, Val * vals, uint64_t nitems, uint64_t * returned_nitems){


   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid != 0) return;

   // if (tile.thread_rank() == 0){

   //    for (int i = 0; i < 10; i++){
   //       printf("%d: %llu, %llu\n", i, keys[i], vals[i]);
   //    }
   // }


   //printf("Starting!\n");

   for (uint64_t i = 0; i < nitems; i++){


      if (!filter->insert(tile, keys[i])){

         if (tile.thread_rank() == 0){

            printf("Inserted %llu / %llu, %f full\n", i, nitems, 1.0*i/nitems);

         }

         returned_nitems[0] = i;

         return;

      } else {

         Val alt_val = 0;
         assert(filter->query(tile, keys[i], alt_val));
         assert(alt_val == vals[i]);


      }

      
   }

   if (tile.thread_rank() == 0) printf("All %llu items inserted\n", nitems);

}



template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;


   if (!filter->insert(tile, keys[tid], vals[tid])){

      if (tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) misses, 1ULL);


   } else{

      Val test_val = 0;
      test_val +=0;
      assert(filter->query(tile, keys[tid], test_val));

      //assert(test_val == vals[tid]);
   }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

template <typename Filter, typename Key, typename Val>
__global__ void debug_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses, bool * missed){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;


   if (!filter->insert(tile, keys[tid], vals[tid])){

      //filter->insert(tile, keys[tid], vals[tid]);

      if (tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) misses, 1ULL);

      missed[tid] = true;


   } else{

      Val test_val = 0;
      assert(filter->query(tile, keys[tid], test_val));

      missed[tid] = false;

      //assert(test_val == vals[tid]);
   }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

template <typename Filter, typename Key, typename Val>
__global__ void debug_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures, bool * missed){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   if (missed[tid]) return;

   Val test_val = 0;
   test_val +=0;

   if (!filter->query(tile,keys[tid], test_val)){


      filter->query(tile,keys[tid], test_val);


      if(tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) query_misses, 1ULL);

   } else {


      // if (test_val != vals[tid] && tile.thread_rank() == 0){
      //    atomicAdd((unsigned long long int *) query_failures, 1ULL);
      // }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_remove_kernel(Filter * filter, Key * keys, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->remove(tile, keys[tid]) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   } 
      //else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));
   // }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

__global__ void count_bf_misses(bool * vals, uint64_t nitems, uint64_t * misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= nitems) return;


   if (!vals[tid]){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   }
}

template <typename Filter, typename Key, typename Val>
__global__ void speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val test_val = 0;

   if (!filter->query(tile,keys[tid], test_val)){


      //filter->query(tile,keys[tid], test_val);


      if(tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) query_misses, 1ULL);

   } else {


      if (test_val != vals[tid] && tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *) query_failures, 1ULL);
      }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Key, typename Val>
__global__ void fp_speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val test_val = 0;

   if (!filter->query(tile,keys[tid], test_val)){


   //    filter->query(tile,keys[tid], test_val);


      if(tile.thread_rank() == 0)
      atomicAdd((unsigned long long int *) query_misses, 1ULL);

   // } else {


      // if (test_val != vals[tid] && tile.thread_rank() == 0){
      //    atomicAdd((unsigned long long int *) query_failures, 1ULL);
      // }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Val>
__host__ void test_tcf_speed(const std::string& filename, int num_bits, int num_batches){


   using Key = uint64_t;
   //using Val = uint8_t;

   //using Filter = tcf;

   //std::cout << "Starting " << filename << " " << num_bits << std::endl;

   // poggers::sizing::size_in_num_slots<2> pre_init ((1ULL << num_bits), (1ULL << num_bits)/100);

   // poggers::sizing::size_in_num_slots<2> * Initializer = &pre_init;

   uint64_t table_nitems = (1ULL << num_bits);

   poggers::sizing::size_in_num_slots<2> pre_init((table_nitems * 90ULL / 100ULL), (table_nitems * 10ULL / 100ULL));
   //poggers::sizing::size_in_num_slots<2> pre_init (table_nitems*.9, table_nitems*.1);

   poggers::sizing::size_in_num_slots<2> * Initializer = &pre_init;



   uint64_t nitems = Initializer->total()*.9;

   Key * host_keys = generate_data<Key>(nitems);
   Val * host_vals = generate_data<Val>(nitems);


   //For MHM TCF - we must clip keys to range [0,2^6);
   //otherwise the fp rate looks super high

   for (uint64_t i = 0; i < nitems; i++){
      host_vals[i] = host_vals[i] % 64;
   }


   Key * fp_keys = generate_data<Key>(nitems);

   Key * dev_keys;

   Val * dev_vals;




   uint64_t * misses;

   hipMallocManaged((void **)& misses, sizeof(uint64_t)*5);
   hipDeviceSynchronize();

   //printf("Data generated\n");

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;

   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   hipDeviceSynchronize();

   //init timing materials
   std::chrono::duration<double>  * insert_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * query_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * fp_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));

   std::chrono::duration<double>  * delete_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));



   uint64_t * batch_amount = (uint64_t *) malloc(num_batches*sizeof(uint64_t));

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);


   for (uint64_t i = 0; i < num_batches; i++){

      uint64_t start_of_batch = i*nitems/num_batches;
      uint64_t items_in_this_batch = (i+1)*nitems/num_batches;

      if (items_in_this_batch > nitems) items_in_this_batch = nitems;

      items_in_this_batch = items_in_this_batch - start_of_batch;


      batch_amount[i] = items_in_this_batch;


      hipMalloc((void **)& dev_keys, items_in_this_batch*sizeof(Key));
      hipMalloc((void **)& dev_vals, items_in_this_batch*sizeof(Val));


      hipMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), hipMemcpyHostToDevice);
      hipMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), hipMemcpyHostToDevice);


      bool * missed;

      hipMalloc((void **)&missed, items_in_this_batch*sizeof(bool));


      uint64_t num_blocks = test_filter->get_num_blocks(items_in_this_batch);
      uint64_t block_size = test_filter->get_block_size(items_in_this_batch);



      //ensure GPU is caught up for next task
      hipDeviceSynchronize();

      auto insert_start = std::chrono::high_resolution_clock::now();

      //add function for configure parameters - should be called by ht and return dim3
      hipLaunchKernelGGL(HIP_KERNEL_NAME(speed_insert_kernel<Filter, Key, Val>), num_blocks, block_size, 0, 0, test_filter, dev_keys, dev_vals, items_in_this_batch, misses);
   

      hipDeviceSynchronize();
      auto insert_end = std::chrono::high_resolution_clock::now();

      insert_diff[i] = insert_end-insert_start;

      hipMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), hipMemcpyHostToDevice);
      hipMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), hipMemcpyHostToDevice);


      hipDeviceSynchronize();

      auto query_start = std::chrono::high_resolution_clock::now();

      hipLaunchKernelGGL(HIP_KERNEL_NAME(speed_query_kernel<Filter, Key, Val>), num_blocks, block_size, 0, 0, test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[1], &misses[2]);
      
      
      hipDeviceSynchronize();
      auto query_end = std::chrono::high_resolution_clock::now();


     
      query_diff[i] = query_end - query_start;

      hipMemcpy(dev_keys, fp_keys+start_of_batch, items_in_this_batch*sizeof(Key), hipMemcpyHostToDevice);
      hipMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), hipMemcpyHostToDevice);


      hipDeviceSynchronize();

      auto fp_start = std::chrono::high_resolution_clock::now();

      hipLaunchKernelGGL(HIP_KERNEL_NAME(speed_query_kernel<Filter, Key, Val>), num_blocks, block_size, 0, 0, test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[3], &misses[4]);


      hipDeviceSynchronize();
      auto fp_end = std::chrono::high_resolution_clock::now();

      fp_diff[i] = fp_end-fp_start;


      hipFree(dev_keys);
      hipFree(dev_vals);

      hipFree(missed);


   }

   //deletes
   // for (uint64_t i = 0; i < num_batches; i++){

   //    uint64_t start_of_batch = i*nitems/num_batches;
   //    uint64_t items_in_this_batch = (i+1)*nitems/num_batches;

   //    if (items_in_this_batch > nitems) items_in_this_batch = nitems;

   //    items_in_this_batch = items_in_this_batch - start_of_batch;


   //   // batch_amount[i] = items_in_this_batch;


   //    hipMalloc((void **)& dev_keys, items_in_this_batch*sizeof(Key));
   //    //hipMalloc((void **)& dev_vals, items_in_this_batch*sizeof(Val));


   //    hipMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), hipMemcpyHostToDevice);
   //    //hipMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), hipMemcpyHostToDevice);

   //    hipDeviceSynchronize();

   //    auto delete_start = std::chrono::high_resolution_clock::now();

   //    hipLaunchKernelGGL(HIP_KERNEL_NAME(speed_query_kernel<Filter, Key, Val>), num_blocks, block_size, 0, 0, test_filter, dev_keys, dev_vals, items_in_this_batch, &misses[1], &misses[2]);
   //    hipDeviceSynchronize();
   //    auto delete_end = std::chrono::high_resolution_clock::now();


     
   //    delete_diff[i] = delete_end - delete_start;

   // }

   hipDeviceSynchronize();


   Filter::free_on_device(test_filter);

   free(host_keys);
   free(host_vals);
   free(fp_keys);

   //free pieces

   //time to output


   printf("nitems: %llu, insert misses: %llu, query missed: %llu, query mismatch %llu, fp missed %llu, fp mismatch %llu\n", nitems, misses[0], misses[1], misses[2], misses[3], misses[4]);

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

  


   const uint64_t scaling_factor = 1000000ULL;



   double insert_throughput = nitems/(scaling_factor*summed_insert_diff.count());
      
   double lookup_throughput = nitems/(scaling_factor*summed_query_diff.count());

   double fp_throughput = nitems/(scaling_factor*summed_fp_diff.count());

   std::cout << "Name,  insert perf (M/s), lookup perf (M/s), FP perf (M/s)" << std::endl;
   std::cout << filename << ": " << insert_throughput << ", " << lookup_throughput << ", " << fp_throughput <<"." << std::endl;

   // std::cout << insert_file << std::endl;
   return;




}



int main(int argc, char** argv) {

   printf("Starting tests\n");

   test_tcf_speed<TCF, uint8_t>("tcf_mhm_20", 20, 20);

   test_tcf_speed<TCF, uint8_t>("tcf_mhm_22", 22, 20);

   test_tcf_speed<TCF, uint8_t>("tcf_mhm_24", 24, 20);

   test_tcf_speed<TCF, uint8_t>("tcf_mhm_26", 26, 20);

   test_tcf_speed<TCF, uint8_t>("tcf_mhm_28", 28, 20);
   

   hipDeviceSynchronize();

   printf("Tests over\n");



   // test_first_fail(22);
   // test_first_fail(24);
   // test_first_fail(26);
   // test_first_fail(28);
   // test_first_fail(30);


   return 0;

}

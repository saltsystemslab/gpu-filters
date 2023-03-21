

Two Choice Filter
_______________________

The Two Choice Filter is a fast approximate data structure based on two-choice-hashing. The base version of the TCF supports insertions, queries, and key-value association. To include the base version of the TCF, include ```<poggers/data_structs/tcf.cuh>``` or add the following template:


```
  using tcf_backing_table = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
  using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tcf_backing_table>;
```

API
___________________

#Host API / Classes

* `sizing_in_num_slots<num_layers>`: This class is used to specify the size and layout of the TCF, 
(uint64_t layer1_num_slots, uint64_t layer2_num_slots)`
* `__host__ static filter_type * generate_on_device(&sizing_in_num_slots<num_layers> sizing)`: Giving a sizing class, initialize the filter on device.
* `__host__ static void free_on_device(filter_type * TCF)`: Free the TCF, handing back the memory to device.
* ` __host__ uint64_t get_num_blocks(uint64_t nitems)`: Given nitems to work with, this gives the number of blocks such that every item has a unique cooperative group.
* `__host__ uint64_t get_block_size(uint64_t nitems_to_insert)`: Get the block size for nitems.



#Device API / Classes

* `__device__ bool insert(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val)`: Uses a cooperative group tile to insert `key` and `value` into the filter.
* `__device__ bool insert_with_delete(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val)`: Insert into the filter, overwriting tombstones if any exist.
* `__device__ bool query(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val & val)`: If the key exists in the filter, return true and store the associated value in `val`.
* `__device__ bool remove(cg::thread_block_tile<Partition_Size> Insert_tile, Key key)`: remove an item from the filter. If the filter is not using `XOR` hash, this can introduce false negatives.
* `__device__ cg::thread_block_tile<Partition_Size> get_my_tile()`: returns the cooperative group associated with this thread. 




TCF Deletes
_______________________

To guarantee that the TCF does not introduce false negatives during deletion, the hash functions used must be depedent. This can be acheived by using XOR power of two hashing, along with linear probing for the backing table (with sufficent probe length, the position of any matching tags will be found by both threads).

An example of deletion behavior can be seen in the test file ```tests/delete_tests.cu```. To include the TCF with deletes, use the following template:

```
using backing_table = poggers::tables::bucketed_table<
    uint64_t, uint16_t,
    poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
        poggers::representations::bit_grouped_container<16, 16>::representation, uint16_t>::representation>::representation,
    4, 8, poggers::insert_schemes::linear_insert_bucket_scheme, 400, poggers::probing_schemes::linearProber,
    poggers::hashers::murmurHasher>;



using del_TCF = poggers::tables::bucketed_table<
    uint64_t, uint16_t,
    poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
        poggers::representations::bit_grouped_container<16, 16>::representation, uint16_t>::representation>::representation,
    4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::XORPowerOfTwoHasher,
    poggers::hashers::murmurHasher, true, backing_table>;

```

In addition, insertions do not check for tombstones by default. To enable this behavior, use ```tcf->insert_with_delete``` instead of ```tcf->insert```. This comes with a small performance hit of 1.1e9 inserts per second instead of 1.3e9.


Building
--------
The TCF components are header only, so linking the underlying library is sufficient to include the TCF.

Building CMake inside of the directory will add tests.


Running Tests
------------

Several tests exist to showcase the behavior/performance of the TCF

* `test_cg_variations`: Iterate over all of the cooperative group options for the primary table and record their throughput.
* `delete_tests`: Test the delete TCF on insertion, query, and deleting 50% of items.
* `sawtooth_test`: Fill the delete TCF to 90% load factor, then delete items in batches.
* `speed_tests`: Test speed of TCF and delete TCF on basic ops.
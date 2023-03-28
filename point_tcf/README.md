

Two Choice Filter (TCF)
----------

The two choice filter (TCF) is a fast approximate data structure based on two-choice hashing. The base version of the TCF supports insertions, queries, and key-value association. To include the base version of the TCF, include ```<poggers/data_structs/tcf.cuh>```


API
----------

Host API / Classes
----------


Wrapper API
* `__host__ static tcf * tcf_wrapper::generate_on_device(uint64_t nslots)`: Given nslots, build a filter with that many slots. Requires the template parameters to be filled.
* `__host__ tcf_wrapper::free_on_device(tcf * tcf_ptr)`: Free the TCF from the wrapper.


Filter Host API
* `__host__ static tcf * generate_on_device(&sizing_in_num_slots<num_layers> sizing)`: Giving a sizing class, initialize the filter on device.
* `__host__ static void free_on_device(tcf * TCF)`: Free the TCF, handing back the memory to device.
* ` __host__ uint64_t get_num_blocks(uint64_t nitems)`: Given nitems to work with, this gives the number of blocks such that every item has a unique cooperative group.
* `__host__ uint64_t get_block_size(uint64_t nitems_to_insert)`: Get the block size for nitems.



Device API / Classes
----------

* `__device__ bool insert(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val)`: Uses a cooperative group tile to insert `key` and `value` into the filter.
* `__device__ bool insert_with_delete(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val val)`: Insert into the filter, overwriting tombstones if any exist.
* `__device__ bool query(cg::thread_block_tile<Partition_Size> Insert_tile, Key key, Val & val)`: If the key exists in the filter, return true and store the associated value in `val`.
* `__device__ bool remove(cg::thread_block_tile<Partition_Size> Insert_tile, Key key)`: remove an item from the filter. If the filter is not using `XOR` hash, this can introduce false negatives.
* `__device__ cg::thread_block_tile<Partition_Size> get_my_tile()`: returns the cooperative group associated with this thread. 




TCF Deletes
----------

Insertions do not replace tombstones by default. To enable this behavior, use ```tcf->insert_with_delete``` instead of ```tcf->insert```. This comes with a small performance hit of 1.1e9 inserts per second instead of 1.3e9.

Configuration
--------------

The TCF Wrapper takes in the following template parameters:

* `key type`: type of input keys. This must be at least `key_bits` bits.
* `value type`: type of input values. This must be at least `val_bits` bits.
* `tag bits`: how many bits are stored per key in the filter.
* `val bits`: how many bits are stored per value in the filter.
* `cooperative group size`: how many threads are assigned per operation.
* `bucket size`: how many keys are stored per bucket.

For example, the tcf with `uint64_t` keys and `uint16_t` values, with `16` bits per key stored as a tag and `16` bits stored per value, with `4` threads per operation and `16` keys per bucket, would be:

`poggers::data_structs::tcf_wrapper<uint64_t, uint16_t, 16, 16, 4, 16>::tcf`.

The template has the following requirements on the input types:

* `sizeof(key_type) >= key_bits`
* `sizeof(val_type) >= val_bits`
* `key_bits+val_bits <= 64`
* `bucket size % cooperative group size == 0`
* `cooperative group size = {1,2,4,8,16,32}`

The TCF uses CUDA's atomicCAS to swap items, so (key,val) pairs are packed into `uint16_t, uint32_t, uint64_t`.


Building
--------
The TCF components are header only, so linking the underlying library is sufficient to include the TCF.

Building CMake inside of the directory will add the tests.


Running Tests
------------

Several tests exist to showcase the behavior/performance of the TCF

* `test_cg_variations`: Iterate over all of the cooperative group options for the and record their throughput on insertions and queries.
* `delete_tests`: Test the TCF on insertions, queries, and deleting 50% of inserted items.
* `sawtooth_test`: Fill the TCF to 90% load factor, then delete items in batches.
* `speed_tests`: Test speed of TCF on basic ops.

* `test_mhm_tcf`: test the configuration of the TCF used in MetaHipMer.


Contributing
------------
Contributions via GitHub pull requests are welcome.


Authors
-------
- Hunter McCoy <hunter@cs.utah.edu>
- Prashant Pandey <pandey@cs.utah.edu>

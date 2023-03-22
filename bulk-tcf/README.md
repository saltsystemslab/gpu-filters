# Bulk TCF


Overview
--------

 The Bulk TCF is a version of the Two Choice Filter to test throughput with bulk-synchronous insertion. The Bulk TCF does not support deletion, although this feature could be supported.

 This design expands on the primary table of the TCF by inserted sorted batches of items into each bucket.

 Groups of buckets are loaded into shared memory, and items choose between two buckets in this local memory block. 

 Two-choice load balancing is done while determining which buckets items belong to. As items are already sorted, this allows for the majority of keys to be inserted directly into the filter in contiguous batched writes, saving IOs.

 This filter is designed to be run in a bulk-synchronous format, where all keys are known during filter construction. The filter does support batched operations but is most efficient when all items can be operated on simultaneously.


API
--------


Template Parameters
--------
The class "bulk_host_tcf" provides a wrapper to make is easy to use the bulk tcf. The api for this wrapper is included below.

template parameters for the filter: `host_bulk_tcf<input_key_type, store_key_type, store_val_type, wrapper>`
* `input_key_type`: type of your input keys
* `store_key_type`: type of keys to be stored in the filter. The size of this determines the false positive rate of the filter.
* `store_value_type`: Values to both be stored and inserted into the filter.
* `wrapper`: To use values add the template `wrapper` from `bulk_tcf_key_val_pair` along with your value type. Otherwise, you can template the filter type without a wrapper as `bulk_host_tcf<input_key_type, store_key_type>`


Metadata
--------

The size of the internal blocks used in the bulk TCF can be modified inside of `include/tcf_bulk_metadata.cuh`. The configurable parameters are:

* `WARPS_PER_BLOCK`: How many warps are assigned per filter block. More warps is more throughput.
* `BLOCKS_PER_THREAD_BLOCK`: how many filter sections are assigned to each thread block. Increasing this increased the maximum load factor the filter can scale to, but the size is limited by shared memory.
* `BYTES_PER_CACHE_LINE / CACHE_LINES_PER_BLOCK`: These two values multiplied together is the amount of space given to each filter section. Upping this increases throughput but is limited by shared memory.


Recommended metadata
--------

For the key-only bulk tcf with key size uint16_t, the optimal configuration is `16, 128, 128, 2`.

For 16 bit keys, 16 bit vals, the optimal configuration is `16, 64, 128, 4`.




Functions
--------
* `__host__ static host_bulk_tcf * build_tcf(uint64_t nslots)`: Construct a filter with `nslosts` slots. This can hold `.85*nslots` items.
*  `__host__ void bulk_insert(Large_Keys * keys, uint64_t nitems, uint64_t * misses)`: Insert `nitems` stored in `keys` into the filter. Counts the number of insertions that fail.
* `__host__ bool * bulk_query(Large_Keys * query_keys, uint64_t nitems)`: Check for Inclusion in the filter. Returns a bitvector where `bit[i]` is set if `query_keys[i]` is found in the filter.
* `__host__ bool * bulk_delete(Large_Keys * delete_keys, uint64_t nitems)`: delete items from the filter. Output is stored in the corresponding hit buffer.

There are also functions for working with values.

*  `__host__ void bulk_insert_values(Large_Keys * keys, Val * vals, uint64_t nitems, uint64_t * misses)`: Insert `nitems` stored in `keys` into the filter. Counts the number of insertions that fail. This variant also stores the value associated with each key.
*  `    __host__ bool * bulk_query_values(Large_Keys * query_keys, Val * output_buffer, uint64_t nitems)`: Query nitems from `query_keys`. Returns a bitvector where `bit[i]` is set if `query_keys[i]` is found. Additionally, `output_buffer[i]` will contain the stored value.



Build
-------
This library depends on Thrust and CUDA.

To build / run:
```bash
 $ mkdir build
 $ cd build
 $ cmake ..
 $ make
 $ ./tcf_wrapper_tests num_bits num_batches
```

Testing
-------

There are currently 3 tests available:

* `tcf_wrapper_tests`: Insert, query, and delete items in batches, recording throughput for all. This test runs using the "bulk_host_tcf" wrapper.
* `sawtooth_test.cu`: Fill the filter to 90% load factor, before removing items in batches. This queries to guarantee that new and old items are visible, to ensure that there are no false negatives.
* `tcf_key_val_tests.cu`: Tests the bulk TCF when configured to use key-val pairs.


Contributing
------------
Contributions via GitHub pull requests are welcome.


Authors
-------
- Hunter McCoy <hunter@cs.utah.edu>
- Prashant Pandey <pandey@cs.utah.edu>

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

The class "bulk_host_tcf" provides a wrapper to make is easy to use the bulk tcf. The api for this wrapper is included below.

* `__host__ static host_bulk_tcf * build_tcf(uint64_t nslots)`: Construct a filter with `nslosts` slots. This can hold `.85*nslots` items.
*  `__host__ void bulk_insert(Large_Keys * keys, uint64_t nitems, uint64_t * misses)`: Insert `nitems` stored in `keys` into the filter. Counts the number of insertions that fail.
* `__host__ bool * bulk_query(Large_Keys * query_keys, uint64_t nitems)`: Check for Inclusion in the filter. Returns a bitvector where `bit[i]` is set if `query_keys[i]` is found in the filter.
* `__host__ bool * bulk_delete(Large_Keys * delete_keys, uint64_t nitems)`: delete items from the filter. Output is stored in the corresponding hit buffer.


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
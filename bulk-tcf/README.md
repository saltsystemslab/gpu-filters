# Bulk TCF

```

Overview
--------

 The Bulk TCF is an experimental version of the Two Choice Filter to test throughput with bulk-synchronous insertion. The Bulk TCF does not support deletion or key-value association, although these features could be supported.

 This design expands on the primary table of the TCF by inserted sorted batches of items into each bucket. Two-choice load balancing is done while determining which buckets items belong to. This allows for the majority of keys to be inserted directly into the filter in contiguous batched writes, saving IOs.
 This filter is designed to be run in a bulk-synchronous format, where all keys are known during filter construction. The filter does support batched insertion but it is not as efficient.


API
--------

* '__host__ uint64_t get_num_blocks()': Used for specifying launch parameters, this returns the number of blocks - valid for the lifetime of the filter.
* '__host__ uint64_t get_num_teams()': Used for specifying launch parameters, this returns the number of teams - valid for the lifetime of the filter.
* '__host__ void attach_lossy_buffers(uint64_t * large_keys, key_type * compressed_keys, uint64_t nitems, uint64_t num_blocks)': prep items to be inserted / queried from the filter. This must be called before insert/query.
* '__host__ void bulk_insert(uint64_t * misses, uint64_t num_teams)': add prepped keys to the filter.
* '__host__ void bulk_query(bool * hits, uint64_t ext_num_teams)': query prepped keys and record misses.


Build
-------
This library depends on Thrust and CUDA.

To build / run:
```bash
 $ make
 $ ./batched_template_tests num_bits num_batches
```

If you have any questions you can reach me at hunter@cs.utah.edu.

Authors
-------
- Hunter McCoy
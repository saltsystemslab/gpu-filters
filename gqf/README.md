# GQF: A Practical Counting Quotient Filter for GPUs


Overview
--------
 The GQF is a general-purpose AMQ that is small and fast, has good
 locality of reference, and supports deletions,
 counting (even on skewed data sets), and highly concurrent
 access. Internally, the GQF is a counting quotient filter, with insert and query schemes modified for high throughput on GPUs.

API
--------

* `__host__ void qf_malloc_device(QF** qf, int nbits, int rbits, int vbits, bool bulk_config)`: Initializes a new GQF with `2^nbits` slots, qf is set to point to the new filter. bulk_config specifies whether or not the system will use locking or bulk inserts. `rbits` is the size of remainder tags stored, `vbits` is size of value bits stored.
* `__host__void qf_destroy_device(QF * qf)`: Free the GQF pointed to by qf.

POINT API
--------


* `__device__ qf_returns point_insert(QF* qf, uint64_t key, uint8_t value, uint8_t flags)`: Insert an ittem into the filter.
* `__device__ qf_returns point_insert_not_exists(QF* qf, uint64_t key, uint8_t value, uint8_t& returnedVal,  uint8_t flags)`: Check if an item is found in the filter. if so, return the item, otherwise, insert it into the filter.
* `__device__ uint64_t point_query(QF* qf, uint64_t key, uint8_t value, uint8_t& returnedVal, uint8_t flags)`: Return the count of an item in the filter, return 0 if the item is not found.
* `__device__ uint64_t point_query_concurrent(QF* qf, uint64_t key, uint8_t value, uint8_t& returnedVal, uint8_t flags)`: Same behavior as point_query, but with locking. Use this when inserts and queries must occur simultaneously and counts are required. (If counts are not necessary, point_insert_not_exists is faster)
* ` __device__ int point_delete(QF* qf, uint64_t key, uint8_t value, uint8_t flags)`: Decrements the count of an item, removing it from the filter if the count ==0. This function locks and is thread-safe. retuns the number of slots freed (-1 for failure).

> `qf_returns` is an enum of either QF_ITEM_FOUND, QF_ITEM_INSERTED, or QF_FULL.


BULK API
--------
* `__host__ void bulk_insert(QF* qf, uint64_t nvals, uint64_t* keys, uint8_t flags)`: Insert a batch of items into the filter using the even-odd insert scheme.
* `__host__ void bulk_insert_reduce(QF* qf, uint64_t nvals, uint64_t* keys, uint8_t flags)`: Insert a batch of items, but perform a reduction before inserting into the CQF. This should be used when the inputs are expected to have heavy skew.
* `__host__ void bulk_query(QF* qf, uint64_t nvals, uint64_t * keys, uint64_t * returns)`: Fills returns with the counts of keys in the filter.
* `__host__ void bulk_delete(QF* qf, uint64_t nvals, uint64_t* keys, uint8_t flags)`: Decrement the counts of all items in keys by one, removing them from the filter if count == 0.


BULK API (with Values)
-------

* `__host__ void bulk_insert_values(QF* qf, uint64_t nvals, uint64_t* keys, uint64_t * vals, uint8_t flags)`: insert a bulk set of key / value pairs. Values will be clipped to be vbits.
* `__host__ void bulk_query_values(QF * qf, uint64_t * keys, uint8_t * vals, bool * hits, uint64_t nvals)`: Retreive value associated with each key. `hits` is a bitvector marking if each item was found.
* `__host__ void bulk_delete_values(QF* qf, uint64_t nvals, uint64_t* keys, uint8_t * vals, uint8_t flags)`: Delete key / value pairs from the filter.

    




Build
-------
This library depends on [Thrust](https://thrust.github.io/). 

In addition, one of the filters available for testing, the SQF, depends on [CUB](https://nvlabs.github.io/cub/) and [ModernGPU](https://moderngpu.github.io/intro.html). 

The code uses two new instructions to implement select on machine words introduced 
in intel's Haswell line of CPUs. However, there is also an alternate implementation
of select on machine words to work on CPUs older than Haswell.

To build:
```bash
 $ source modules.sh
 $ make clean && make
```


The argument to -n is the log of the number of slots in the GQF. For example,
 to create a GQF with 2^30 slots, the argument will be -n 30.

The argument to -d is the filter being tested. The currently supported filter types are:

 - gqf (GQF bulk API)
 - point (GQF point API)
 - sqf (Standard Quotient Filter from [Geil et al.](https://escholarship.org/uc/item/3v12f7dn))
 - rsqf (Rank-Select Quotient Filter from [Geil et al.](https://escholarship.org/uc/item/3v12f7dn))
 - bloom (Bloom Filter)


To set the number of bits stored per (key,value) pair in the GQF, change the `QF_BITS_PER_SLOT` definition on line 30 of `gqf_int.cuh`. The default is 8.


Testing
-------
test.cu contains the tests used to compare the various filters
gqf_verify.cu tests the correctness of the GQF on skewed, zipfian, and kmer (FASTQ) datasets, and verifies the correctness of operations like counting and deletions.

options for test
```bash
 $ ./test -d [gqf,point,sqf,rsqf, bloom] -n [numbits] -v [1 for verbose] -o [outputfile - three files with extensions -inserts.txt, -exists-lookup.txt, -false-lookup.txt] -p [npoints] -f [1 for false-positive reporting]
```

To generate the false positive rate for a filter, add the flag `-f`. This calculates the false positive rate per batch, so to see the false positive rate of the filter you must insert all items in one batch. To do this, set the number of points to 1 and the buffer size to be nitems. For example, `./test -n 24 -b 24 -p 1 -f` will print out the false-positive rate of the GQF.


options for gqf_verify
```bash
 $ ./gqf_verify [nbits] [0 bulk, 1 reduce] [0 random, 1 file, 2 random copies, 3 fastq, 4 zipfian] [filename if previous arg is 1 or 3]
```

Example Runs
-------------

```bash
 $ ./test -d gqf -n 28 -p 20 -o results/gqf/28
```
This will test the gqf filter with a dataset of 2^28 items split into 20 batches, with items inserted/queried via bulk methods and the results written into results/gqf/28-inserts.txt, results/gqf/28-exists-lookup.txt, and results/gqf/28-false-lookup.txt.


```bash
 $ ./gqf_verify 26 1 4
```
This will test the gqf on a dataset of 2^26 bits, where the data is generated as a zipfian distribution and the items are inserted using a reduction scheme. If running the reduction test, you must set the CudaToolKit version to `<= 10.6`: A change in more recent versions of Thrust causes a warp misaligned issue.

Artifact Descriptions
----------------------

As part of the artifacts for the sc22 submission, this repo contains the files necessary to rebuild the graphs from our submission. While additional info is available in the additional artifact sumbission document, running

```bash
 $ ./scripts/generate_data.sh
```
will produce a pdf document containing figures identical to those used in our submission, and with results generated from your machine.


Contributing
------------
Contributions via GitHub pull requests are welcome.


Authors
-------
- Hunter McCoy <hunter@cs.utah.edu>
- Prashant Pandey <pandey@cs.utah.edu>

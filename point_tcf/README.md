

Two Choice Filter
_______________________

The Two Choice Filter is a fast approximate data structure based on two-choice-hashing. The base version of the TCF supports insertions, queries, and key-value association. To include the base version of the TCF, include ```<poggers/data_structs/tcf.cuh>``` or add the following template:


```
  using tcf_backing_table = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
  using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tcf_backing_table>;
```


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



API
___________________





PUBLICATIONS
______________

The TCF was released in the publication "High Performance Filters For GPUs" at PPOPP 2023!

The paper is available here: https://dl.acm.org/doi/pdf/10.1145/3572848.3577507

To cite the TCF, please use the following citation:
```
@inproceedings{10.1145/3572848.3577507,
author = {McCoy, Hunter and Hofmeyr, Steven and Yelick, Katherine and Pandey, Prashant},
title = {High-Performance Filters for GPUs},
year = {2023},
isbn = {9798400700156},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3572848.3577507},
doi = {10.1145/3572848.3577507},
abstract = {Filters approximately store a set of items while trading off accuracy for space-efficiency and can address the limited memory on accelerators, such as GPUs. However, there is a lack of high-performance and feature-rich GPU filters as most advancements in filter research has focused on CPUs.In this paper, we explore the design space of filters with a goal to develop massively parallel, high performance, and feature rich filters for GPUs. We evaluate various filter designs in terms of performance, usability, and supported features and identify two filter designs that offer the right trade off in terms of performance, features, and usability.We present two new GPU-based filters, the TCF and GQF, that can be employed in various high performance data analytics applications. The TCF is a set membership filter and supports faster inserts and queries, whereas the GQF supports counting which comes at an additional performance cost. Both the GQF and TCF provide point and bulk insertion API and are designed to exploit the massive parallelism in the GPU without sacrificing usability and necessary features. The TCF and GQF are up to 4.4\texttimes{} and 1.4\texttimes{} faster than the previous GPU filters in our benchmarks and at the same time overcome the fundamental constraints in performance and usability in current GPU filters.},
booktitle = {Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
pages = {160–173},
numpages = {14},
location = {Montreal, QC, Canada},
series = {PPoPP '23}
}
```
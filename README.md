# GPU Filters
High-Performance GPU Filters

FEATURES
________________

1) Two Choice Filter (TCF)
	- A set of high-performance template components for building point key-value data strucutures.
	- Modular components to specify data layout, access patterns, and false-positive rate.
    - Recursive template structure lets you build efficient multi-layer tables with minimal effort.

2) Bulk Two Choice Filter (Bulk TCF)
	- A TCF implementation optimized for bulk data. Uses Thrust to efficiently group keys for operations.


3) GPU Quotient Filter (GQF)
	- A Counting Quotient Filter implementation on the GPU
	- supports insertions, queries, deletions, key-value assocation, and counting.
	- Point and Bulk API for different workloads
	- reduction insertion scheme for Zipfian workloads.



____________________

To add these filters to a project, you can use the [CMake Package Manager](https://github.com/cpm-cmake/CPM.cmake)


To add CPM, add 

```include(cmake/CPM.cmake)``` 

to your cmake file.

To add poggers, include the following snippet and select a version.

If you remove the version tag, CPM will pull the most up-to-date build.

```
CPMAddPackage(
  NAME gpu-filters
  GITHUB_REPOSITORY saltsystemlab/gpu-filters
  GIT_TAG origin/main
)
```

To cache the library, specify a download folder:

```set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/downloaded_libraries")```



Building Tests
___________________

There are a series of optional tests that can be included with the build.

To build the tests, specify the ```-DBUILD_TESTS=ON``` flag to CMake.


FUTURE WORK
________________

2) Dynamic GPU Allocation - In Progress!
3) Host-Device pinned device-side communication
4) Sparse ML components.


AUTHORS
______________



PUBLICATIONS
______________

These filters were released in the publication "High Performance Filters For GPUs" at PPOPP 2023.

The paper is available here: https://dl.acm.org/doi/pdf/10.1145/3572848.3577507

To cite these filters, please use the following citation:
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


PULL REQUESTS
______________


CONTACT
_____________

If you have any questions/comments/concerns, you can reach contact us at:

	Hunter McCoy: hunter@cs.utah.edu.
#ifndef METADATA
#define METADATA


#define DEBUG_ASSERTS 0

#define DELETE_DEBUG_ASSERTS 0

#define DESTRUCTIVE_CHECK 0
//#define MAX_FILL 28
#define SINGLE_REGION 0

//do blocks assume exclusive access? if yes, no need to lock
//this is useful for batched scenarios.
#define EXCLUSIVE_ACCESS 1


//number of warps launched per grid block
// #define WARPS_PER_BLOCK 16
// #define BLOCK_SIZE (WARPS_PER_BLOCK * 32)

// #define BLOCKS_PER_THREAD_BLOCK 128

#define WARPS_PER_BLOCK 16
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)

#define BLOCKS_PER_THREAD_BLOCK 128

//# of blocks to be inserted per warp in the bulked insert phase
//#define REGIONS_PER_WARP 8


//power of 2 metadata
//#define POWER_BLOCK_SIZE 1024
//#define TOMBSTONE 1000000000000ULL
#define TOMBSTONE_VAL 0

#define SLOTS_PER_CONST_BLOCK 32



//Atomic blocks stats

#define TAG_BITS 16
#define VAL_BITS 16 

#define BYTES_PER_CACHE_LINE 128
#define CACHE_LINES_PER_BLOCK 2


#endif
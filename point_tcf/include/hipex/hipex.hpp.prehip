#include <cooperative_groups.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>

using namespace cooperative_groups;
#define GRPSIZE 32
namespace hipex{

    __device__ uint32_t reduce(cooperative_groups::thread_group &g, uint32_t* target){
        
        int lane = (int)g.thread_rank();
        uint32_t val = target[lane];
        for(int i = g.size()/2; i > 0; i /=2){
            target[lane] = val;
            g.sync();
            if(lane < i)
                val |= target[lane+i]; 
            g.sync();
        }
        if(lane == 0)
            target[lane] = val;
        g.sync();
    }

    __device__ uint32_t ohc_id(uint32_t idx){
        uint32_t ohc_ = 0;
        ohc_ |= 1U << idx;

        return ohc_;
    }

    // template<Typename T> make it templated later, may be only uptill 32 buts for now
    __device__ uint32_t ballot(cooperative_groups::thread_group &g, int predicate){
        uint32_t ballot_ = 0;
        __shared__ uint32_t pred_sh[GRPSIZE]; // thread group sizes are limited to 32 for now
        
        assert(g.size() <= GRPSIZE);

        if(g.thread_rank() < GRPSIZE){
            pred_sh[g.thread_rank()] = 0;
            if(predicate != 0)
                pred_sh[g.thread_rank()] = ohc_id(g.thread_rank());
        }
        g.sync();
        reduce(g, pred_sh);
        ballot_ = pred_sh[0];
        return ballot_;
    }

    __device__ uint32_t meta_group_size(cooperative_groups::thread_group &g){
        return (blockDim.x/g.size());
    }

    __device__ uint32_t meta_group_rank(cooperative_groups::thread_group &g){
       return (threadIdx.x/g.size());
    }

}
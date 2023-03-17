TARGETS=batched_template_tests

ifdef D
	DEBUG=-g -G
	OPT=
else
	DEBUG=
	OPT=-O3
endif

ifdef NH
	ARCH=
else
	ARCH=-msse4.2 -D__SSE4_2_
endif

ifdef P
	PROFILE=-pg -no-pie # for bug in gprof.
endif

LOC_INCLUDE=include
LOC_SRC=src
LOC_TEST=test
OBJDIR=obj



CC = gcc
CXX = g++ -std=c++20
CU = nvcc -dc -x cu
LD = nvcc

CXXFLAGS = -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -Iinclude 

CUFLAGS = $(DEBUG) $(OPT) -arch=sm_70 -rdc=true -I. -Iinclude -lineinfo

CUDALINK = -L/usr/common/software/sles15_cgpu/cuda/11.1.1/lib64/compat -L/usr/common/software/sles15_cgpu/cuda/11.1.1/lib64 -L/usr/common/software/sles15_cgpu/cuda/11.1.1/extras/CUPTI/lib6 -lcurand --nvlink-options -suppress-stack-size-warning

LDFLAGS = $(DEBUG) $(PROFILE) $(OPT) $(CUDALINK) -arch=sm_70 -lpthread -lssl -lcrypto -lm -lcuda -lcudart -lgomp


#
# declaration of dependencies
#

all: $(TARGETS)

# dependencies between programs and .o files

data_gen:						$(OBJDIR)/data_gen.o

test:							$(OBJDIR)/test.o \
								$(OBJDIR)/vqf_block.o

team_test:							$(OBJDIR)/team_test.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

gpu_block_test:						$(OBJDIR)/gpu_block_test.o \
								$(OBJDIR)/gpu_block.o \
								$(OBJDIR)/warp_utils.o

vqf_tests:						$(OBJDIR)/vqf_tests.o \
								$(OBJDIR)/vqf.o \
								$(OBJDIR)/vqf_block.o

team_vqf_tests:						$(OBJDIR)/team_vqf_tests.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o


sort_vqf_tests:						$(OBJDIR)/sort_vqf_tests.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

vqf_correctness_check:				$(OBJDIR)/vqf_correctness_check.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

locking_vqf_test:						$(OBJDIR)/locking_vqf_test.o \
								$(OBJDIR)/vqf.o \
								$(OBJDIR)/vqf_block.o


small_vqf_tests:						$(OBJDIR)/small_vqf_tests.o \
								$(OBJDIR)/team_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o

mega_vqf_tests:						$(OBJDIR)/mega_vqf_tests.o \
								$(OBJDIR)/mega_vqf.o \
								$(OBJDIR)/megablock.o \
								$(OBJDIR)/warp_utils.o

power_of_one_test:					$(OBJDIR)/power_of_one_test.o \
								$(OBJDIR)/single_vqf.o \
								$(OBJDIR)/vqf_team_block.o \
								$(OBJDIR)/warp_utils.o \
								$(OBJDIR)/hashutil.o

optimized_vqf_tests:				$(OBJDIR)/optimized_vqf_tests.o \
								$(OBJDIR)/optimized_vqf.o \
								$(OBJDIR)/warp_utils.o \
								$(OBJDIR)/hashutil.o \
								$(OBJDIR)/gpu_block.o


power_of_two_test:				$(OBJDIR)/power_of_two_test.o \
								$(OBJDIR)/optimized_vqf.o \
								$(OBJDIR)/warp_utils.o \
								$(OBJDIR)/hashutil.o \
								$(OBJDIR)/gpu_block.o


block_vqf_tests:			$(OBJDIR)/block_vqf_tests.o \
								$(OBJDIR)/block_vqf.o \
								$(OBJDIR)/warp_utils.o \
								$(OBJDIR)/hashutil.o \
								$(OBJDIR)/gpu_block.o

shared_test:					$(OBJDIR)/shared_test.o


atomic_tests:				$(OBJDIR)/atomic_tests.o \
							$(OBJDIR)/atomic_vqf.o \
							$(OBJDIR)/atomic_block.o \
							$(OBJDIR)/hashutil.o \
							$(OBJDIR)/sorting_helper.o


large_sort_tests:			$(OBJDIR)/large_sort_tests.o 

multi_vqf_tests: 			$(OBJDIR)/multi_vqf_tests.o \
							$(OBJDIR)/multi_vqf_host.o \
							$(OBJDIR)/atomic_vqf.o \
							$(OBJDIR)/atomic_block.o \
							$(OBJDIR)/hashutil.o


sorting_tests:				$(OBJDIR)/sorting_tests.o \
							$(OBJDIR)/sorting_helper.o


sorted_block_vqf_tests:		$(OBJDIR)/sorted_block_vqf_tests.o \
							$(OBJDIR)/sorted_block_vqf.o \
							$(OBJDIR)/atomic_block.o \
							$(OBJDIR)/hashutil.o \
							$(OBJDIR)/sorting_helper.o

batched_vqf_tests:		$(OBJDIR)/batched_vqf_tests.o \
							$(OBJDIR)/sorted_block_vqf.o \
							$(OBJDIR)/atomic_block.o \
							$(OBJDIR)/hashutil.o \
							$(OBJDIR)/sorting_helper.o \



batched_hash_table_tests: 	$(OBJDIR)/batched_hash_table_tests.o \
							$(OBJDIR)/gpu_quad_hash_table.o \
							$(OBJDIR)/hashutil.o


batch_global_load_vqf_tests:	$(OBJDIR)/batch_global_load_vqf_tests.o \
								$(OBJDIR)/warp_storage_block.o \
								$(OBJDIR)/global_load_vqf.o \
								$(OBJDIR)/hashutil.o

templated_test:					$(OBJDIR)/templated_test.o \
								$(OBJDIR)/hashutil.o

batched_template_tests:			$(OBJDIR)/batched_template_tests.o \
								$(OBJDIR)/hashutil.o

persistent_batched_template_tests:			$(OBJDIR)/persistent_batched_template_tests.o \
								$(OBJDIR)/hashutil.o


cuda_queue_tests:				$(OBJDIR)/cuda_queue_tests.o


const_batched_template_tests:			$(OBJDIR)/const_batched_template_tests.o \
								$(OBJDIR)/hashutil.o
# dependencies between .o files and .cc (or .c) files


#$(OBJDIR)/RSQF.o: $(LOC_SRC)/RSQF.cu $(LOC_INCLUDE)/RSQF.cuh

$(OBJDIR)/vqf_block.o: $(LOC_SRC)/vqf_block.cu $(LOC_INCLUDE)/vqf_block.cuh
$(OBJDIR)/vqf.o: $(LOC_SRC)/vqf.cu $(LOC_INCLUDE)/vqf.cuh $(LOC_SRC)/vqf_block.cu $(LOC_INCLUDE)/vqf_block.cuh
$(OBJDIR)/vqf_team_block.o: $(LOC_SRC)/vqf_team_block.cu $(LOC_INCLUDE)/vqf_team_block.cuh
$(OBJDIR)/warp_utils.o: $(LOC_SRC)/warp_utils.cu $(LOC_INCLUDE)/warp_utils.cuh
$(OBJDIR)/team_vqf.o: $(LOC_SRC)/team_vqf.cu $(LOC_INCLUDE)/team_vqf.cuh $(LOC_SRC)/vqf_team_block.cu $(LOC_INCLUDE)/vqf_team_block.cuh
$(OBJDIR)/megablock.o: $(LOC_SRC)/megablock.cu $(LOC_INCLUDE)/megablock.cuh
$(OBJDIR)/mega_vqf.o: $(LOC_SRC)/mega_vqf.cu $(LOC_INCLUDE)/mega_vqf.cuh $(LOC_SRC)/megablock.cu $(LOC_INCLUDE)/megablock.cuh
$(OBJDIR)/single_vqf.o: $(LOC_SRC)/single_vqf.cu $(LOC_INCLUDE)/single_vqf.cuh $(LOC_SRC)/vqf_team_block.cu $(LOC_INCLUDE)/vqf_team_block.cuh $(LOC_INCLUDE)/metadata.cuh
$(OBJDIR)/hashutil.o: $(LOC_SRC)/hashutil.cu $(LOC_INCLUDE)/hashutil.cuh
$(OBJDIR)/optimized_vqf.o: $(LOC_SRC)/optimized_vqf.cu $(LOC_INCLUDE)/optimized_vqf.cuh $(LOC_INCLUDE)/metadata.cuh
$(OBJDIR)/gpu_block.o: $(LOC_SRC)/gpu_block.cu $(LOC_INCLUDE)/gpu_block.cuh
$(OBJDIR)/block_vqf.o: $(LOC_SRC)/block_vqf.cu $(LOC_INCLUDE)/block_vqf.cuh $(LOC_INCLUDE)/metadata.cuh
$(OBJDIR)/atomic_block.o: $(LOC_SRC)/atomic_block.cu $(LOC_INCLUDE)/atomic_block.cuh $(LOC_INCLUDE)/metadata.cuh
$(OBJDIR)/atomic_vqf.o: $(LOC_SRC)/atomic_vqf.cu $(LOC_INCLUDE)/atomic_vqf.cuh $(LOC_INCLUDE)/metadata.cuh
$(OBJDIR)/multi_vqf_host.o: $(LOC_SRC)/multi_vqf_host.cu $(LOC_INCLUDE)/multi_vqf_host.cuh $(LOC_INCLUDE)/metadata.cuh 
$(OBJDIR)/sorting_helper.o: $(LOC_SRC)/sorting_helper.cu $(LOC_INCLUDE)/sorting_helper.cuh
$(OBJDIR)/atomic_vqf.o: $(LOC_SRC)/sorted_block_vqf.cu $(LOC_INCLUDE)/sorted_block_vqf.cuh $(LOC_INCLUDE)/metadata.cuh
$(OBJDIR)/gpu_quad_hash_table.o: $(LOC_SRC)/gpu_quad_hash_table.cu $(LOC_INCLUDE)/gpu_quad_hash_table.cuh
$(OBJDIR)/warp_storage_block.o: $(LOC_SRC)/warp_storage_block.cu $(LOC_INCLUDE)/warp_storage_block.cuh
$(OBJDIR)/global_load_vqf.o: $(LOC_SRC)/global_load_vqf.cu $(LOC_INCLUDE)/global_load_vqf.cuh
$(OBJDIR)/templated_block.o: $(LOC_SRC)/templated_block.cu $(LOC_INCLUDE)/templated_block.cuh

#
# generic build rules -- add -Xcompiler -fopenmp to .cu to add back in mpi
#

$(TARGETS):
	$(LD) $^ -o $@ $(LDFLAGS)


$(OBJDIR)/%.o: $(LOC_SRC)/%.cu | $(OBJDIR)
	$(CU) $(CUFLAGS) $(INCLUDE) -dc $< -o $@ 




$(OBJDIR)/%.o: $(LOC_SRC)/%.cc | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -c -o $@ 

$(OBJDIR)/%.o: $(LOC_SRC)/%.c | $(OBJDIR)
	$(CC) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

$(OBJDIR)/%.o: $(LOC_TEST)/%.c | $(OBJDIR)
	$(CC) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGETS) core

TARGETS=test gqf_verify sawtooth_test

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



CC = gcc -std=gnu11
CXX = g++ -std=c++11
CU = nvcc -dc -x cu
LD = nvcc

CXXFLAGS = -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -Iinclude

CUFLAGS = $(DEBUG) -arch=sm_70 -rdc=true -I. -Iinclude

CUDALINK = -L/usr/common/software/sles15_cgpu/cuda/11.1.1/lib64/compat -L/usr/common/software/sles15_cgpu/cuda/11.1.1/lib64 -L/usr/common/software/sles15_cgpu/cuda/11.1.1/extras/CUPTI/lib6 -lcurand --nvlink-options -suppress-stack-size-warning

LDFLAGS = $(DEBUG) $(PROFILE) $(OPT) $(CUDALINK) -arch=sm_70 -lpthread -lssl -lcrypto -lm -lcuda -lcudart


#
# declaration of dependencies
#

all: $(TARGETS)

# dependencies between programs and .o files

cluster_length_test:					$(OBJDIR)/cluster_length_test.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/hashutil.o \
										$(OBJDIR)/partitioned_counter.o


gqf_verify:								$(OBJDIR)/gqf_verify.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/hashutil.o \
										$(OBJDIR)/partitioned_counter.o \
										$(OBJDIR)/zipf.o


test:							$(OBJDIR)/test.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/hashutil.o \
										$(OBJDIR)/partitioned_counter.o \
										$(OBJDIR)/RSQF.o \
										$(OBJDIR)/sqf.o \
										$(OBJDIR)/zipf.o


sawtooth_test:								$(OBJDIR)/sawtooth_test.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/hashutil.o \
										$(OBJDIR)/partitioned_counter.o \

# dependencies between .o files and .h files



$(OBJDIR)/cluster_length_test.o: 			$(LOC_INCLUDE)/gqf.cuh \
															$(LOC_INCLUDE)/hashutil.cuh \
															$(LOC_INCLUDE)/partitioned_counter.cuh

$(OBJDIR)/gqf_verify.o: 						$(LOC_INCLUDE)/gqf.cuh \
															$(LOC_INCLUDE)/hashutil.cuh \
															$(LOC_INCLUDE)/partitioned_counter.cuh


$(OBJDIR)/test.o:								$(LOC_INCLUDE)/gqf_wrapper.cuh \
															$(LOC_INCLUDE)/partitioned_counter.cuh \
															$(LOC_INCLUDE)/cu_wrapper.cuh \
															$(LOC_INCLUDE)/RSQF.cuh \
															$(LOC_INCLUDE)/sqf.cuh \



$(OBJDIR)/sawtooth_test.o: 						$(LOC_INCLUDE)/gqf.cuh \
															$(LOC_INCLUDE)/hashutil.cuh \
															$(LOC_INCLUDE)/partitioned_counter.cuh




# dependencies between .o files and .cc (or .c) files


$(OBJDIR)/RSQF.o: $(LOC_SRC)/RSQF.cu $(LOC_INCLUDE)/RSQF.cuh
$(OBJDIR)/gqf.o:							$(LOC_SRC)/gqf.cu $(LOC_INCLUDE)/gqf.cuh
$(OBJDIR)/hashutil.o:					$(LOC_SRC)/hashutil.cu $(LOC_INCLUDE)/hashutil.cuh
$(OBJDIR)/partitioned_counter.o:	$(LOC_INCLUDE)/partitioned_counter.cuh
$(OBJDIR)/sqf.o: $(LOC_SRC)/sqf.cu $(LOC_INCLUDE)/sqf.cuh
$(OBJDIR)/zipf.o: $(LOC_SRC)/zipf.cu $(LOC_INCLUDE)/zipf.cuh

#
# generic build rules
#

$(TARGETS):
	$(LD) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/sqf.o: $(LOC_SRC)/sqf.cu | $(OBJDIR)
	$(CU) $(CUFLAGS) $(INCLUDE) --extended-lambda -dc $< -o $@

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

// sve-demo.c

#include <errno.h>
#include <memory>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#include <arm_sve.h>
#endif

#include <Runtime.h>

bool gKeepRunning = true;
bool gHasErrors = false;

static void handle_process_exit_signal(int signo) {
  (void)signo;
  gKeepRunning = false;
}

std::vector<int32_t> makeVector(size_t size) {
  std::vector<int32_t> result;
  result.resize(size);

  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = rand();
    if (rand() % 2 > 0) {
      result[i] *= -1;
    }
  }

  return result;
}

bool compareVectors(const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  if (va.size() != vb.size()) {
    return false;
  }

  for (size_t i = 0; i < va.size(); ++i) {
    if (va[i] != vb[i]) {
      return false;
    }
  }

  return true;
}

void vector_print_i32_c(const int threadIdx, const char* const name, const std::vector<int32_t>& v) {
  const size_t vPrintSize = std::min(v.size(), (size_t)50);

  printf("[t=%d] %s[%llu]={", threadIdx, name, (long long unsigned int)v.size());
  for (size_t i = 0; i < vPrintSize; ++i) {
    printf("%ld,", (long int)v[i]);
  }
  if (vPrintSize != v.size()) {
    printf("...");
  }
  printf("}\n");
}

void __attribute__ ((noinline)) vector_add_i32_c(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  if (vres.size() != va.size() || vres.size() != vb.size()) {
    return;
  }

  for (size_t i = 0; i < va.size(); ++i) {
    vres[i] = va[i] + vb[i];
  }
}

void __attribute__ ((noinline)) vector_add_i32_c_prefetch(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  if (vres.size() != va.size() || vres.size() != vb.size()) {
    return;
  }

  __builtin_prefetch(&va[0], 0, 0);
  __builtin_prefetch(&vb[0], 0, 0);

  for (size_t i = 0; i < va.size(); ++i) {
  __builtin_prefetch(&va[i+1], 0, 0);
  __builtin_prefetch(&vb[i+1], 0, 0);

    vres[i] = va[i] + vb[i];
  }
}

#if defined(__aarch64__)
void __attribute__ ((noinline)) vector_add_i32_neon(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  if (vres.size() != va.size() || vres.size() != vb.size()) {
    return;
  }

  size_t len = va.size();
  const int32_t* vaItr = &va[0];
  const int32_t* vbItr = &vb[0];
  int32_t* resultItr = &vres[0];

  while (len >= 4) {
    *((int32x4_t*)resultItr) = vaddq_s32(*((int32x4_t*)vaItr), *((int32x4_t*)vbItr));

    len -= 4;
    vaItr += 4;
    vbItr += 4;
    resultItr += 4;
  }

  while (len > 0) {
    *resultItr = *vaItr + *vbItr;

    len--;
    vaItr++;
    vbItr++;
    resultItr++;
  }
}

void __attribute__ ((noinline)) vector_add_i32_sve(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  const uint32_t arrSize = vres.size();

  if (arrSize != va.size() || arrSize != vb.size()) {
    return;
  }

  const uint32_t kIdxStep = svcntw();

  int32_t* const vresBase = &vres[0];
  const int32_t* const vaBase = &va[0];
  const int32_t* const vbBase = &vb[0];

  uint32_t idx = 0;
  svbool_t pred = svwhilelt_b32(idx, arrSize);
  svint32_t sva, svb, svres;

  while (svptest_first(svptrue_b32(), pred)) {
    // load operands
    sva = svld1_s32(pred, &vaBase[idx]);
    svb = svld1_s32(pred, &vbBase[idx]);

    // add, x = undefined behavior for non-selected elements (fastest)
    svres = svadd_s32_x(pred, sva, svb);

    // store result
    svst1_s32(pred, &vresBase[idx], svres);

    // update index and pred
    idx += kIdxStep;
    pred = svwhilelt_b32(idx, arrSize);
  }
}

void __attribute__ ((noinline)) vector_add_i32_sve_prefetch(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  const uint32_t arrSize = vres.size();

  if (arrSize != va.size() || arrSize != vb.size()) {
    return;
  }

  const uint32_t kIdxStep = svcntw();

  int32_t* const vresBase = &vres[0];
  const int32_t* const vaBase = &va[0];
  const int32_t* const vbBase = &vb[0];

  // prefetch hint for first 3 chunks of input vectors
  svprfw(svptrue_b32(),   vaBase,                 SV_PLDL1STRM);
  svprfw(svptrue_b32(),   vaBase + kIdxStep,      SV_PLDL1STRM);
  svprfw(svptrue_b32(),   vaBase + (2*kIdxStep),  SV_PLDL1STRM);
  svprfw(svptrue_b32(),   vbBase,                 SV_PLDL1STRM);
  svprfw(svptrue_b32(),   vbBase + kIdxStep,      SV_PLDL1STRM);
  svprfw(svptrue_b32(),   vbBase + (2*kIdxStep),  SV_PLDL1STRM);

  const int32_t* vaPrefetchBase = vaBase + (3*kIdxStep);
  const int32_t* vbPrefetchBase = vbBase + (3*kIdxStep);

  uint32_t idx = 0;
  svbool_t pred = svwhilelt_b32(idx, arrSize);
  svint32_t sva, svb, svres;

  while (svptest_first(svptrue_b32(), pred)) {
    // load operands
    sva = svld1_s32(pred, &vaBase[idx]);
    svb = svld1_s32(pred, &vbBase[idx]);

    // prefetch hint for next element
    // TODO This is actually slower because gcc doesn't use the right
    //  overload of PRFW instruction. May need to rewrite in pure asm.
    svprfw(svptrue_b32(), (vaPrefetchBase + idx), SV_PLDL1STRM);
    svprfw(svptrue_b32(), (vbPrefetchBase + idx), SV_PLDL1STRM);

    // add, x = undefined behavior for non-selected elements (fastest)
    svres = svadd_s32_x(pred, sva, svb);

    // store result
    svst1_s32(pred, &vresBase[idx], svres);

    // update index and pred
    idx += kIdxStep;
    pred = svwhilelt_b32(idx, arrSize);
  }
}

void __attribute__ ((noinline)) vector_add_i32_sve_raw(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  const uint32_t arrSize = vres.size();

  if (arrSize != va.size() || arrSize != vb.size()) {
    return;
  }

  int32_t* const vresBase = &vres[0];
  const int32_t* const vaBase = &va[0];
  const int32_t* const vbBase = &vb[0];

  uint32_t idx = 0;

  const int32_t kIdxStep = svcntw();

  // pN.s = predicate reg N, s = 32-bit size words
  asm (
    // WHILELO is unsigned version of WHILELT
    "WHILELO p0.s, %[idx], %[arrSize]; "
    "B.NONE L_loop_end_%=; "

    "L_loop_begin_%=: "

    "LD1W z0.s, p0/Z, [%[vaBase], %[idx], LSL #2]; "
    "LD1W z1.s, p0/Z, [%[vbBase], %[idx], LSL #2]; "

    // Optimization: Unconditional vector add seems a bit faster than predicated add,  based on GCC -Ofast loop unroll.
    // OLD: ADD  z0.s, p0/M, z0.s, z1.s
    "ADD z0.s, z0.s, z1.s; "

    "ST1W z0.s, p0, [%[vresBase], %[idx], LSL #2]; "

    // Optimization: Simple ADD seems a bit faster than INCW,  based on GCC -Ofast loop unroll.
    // OLD: INCW %[idx]
    "ADD %[idx], %[idx], %[kIdxStep]; "

    "WHILELO p0.s, %[idx], %[arrSize]; "
    "B.FIRST L_loop_begin_%=; "

    "L_loop_end_%=: "
    : // no output
    :
      [arrSize] "r" (arrSize),
      [vresBase] "r" (vresBase),
      [vaBase] "r" (vaBase),
      [vbBase] "r" (vbBase),
      [idx] "r" (idx),
      [kIdxStep] "r" (kIdxStep)
    : "p0", "z0", "z1"
  );
}

void __attribute__ ((noinline)) vector_add_i32_sve_prefetch_raw(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
  const uint32_t arrSize = vres.size();

  if (arrSize != va.size() || arrSize != vb.size()) {
    return;
  }

  int32_t* const vresBase = &vres[0];
  const int32_t* const vaBase = &va[0];
  const int32_t* const vbBase = &vb[0];

  const int32_t* vaPrefetchBase = 0;
  const int32_t* vbPrefetchBase = 0;

  uint32_t idx = 0;

  const int32_t kIdxStep = svcntw();

  // pN.s = predicate reg N, s = 32-bit size words
  asm (
    "PTRUE   p1.s; "

    // prefect hint for first 3 chunks of input vectors
    "PRFW PLDL1STRM, p1,[%[vaBase], #0, MUL VL]; "
    "PRFW PLDL1STRM, p1,[%[vaBase], #1, MUL VL]; "
    "PRFW PLDL1STRM, p1,[%[vaBase], #2, MUL VL]; "
    "PRFW PLDL1STRM, p1,[%[vbBase], #0, MUL VL]; "
    "PRFW PLDL1STRM, p1,[%[vbBase], #1, MUL VL]; "
    "PRFW PLDL1STRM, p1,[%[vbBase], #2, MUL VL]; "

    // adjust prefetch base ptr for subsequent operations
    "ADDVL %[vaPrefetchBase], %[vaBase], #3; "
    "ADDVL %[vbPrefetchBase], %[vbBase], #3; "

    // WHILELO is unsigned version of WHILELT
    "WHILELO p0.s, %[idx], %[arrSize]; "
    "B.NONE L_loop_end_%=; "

    "L_loop_begin_%=: "

    "LD1W z0.s, p0/Z, [%[vaBase], %[idx], LSL #2]; "
    "LD1W z1.s, p0/Z, [%[vbBase], %[idx], LSL #2]; "

    // prefetch hint for next element
    "PRFW PLDL1STRM, p1,[%[vaPrefetchBase], %[idx], LSL #2]; "
    "PRFW PLDL1STRM, p1,[%[vbPrefetchBase], %[idx], LSL #2]; "

    // Optimization: Unconditional vector add seems a bit faster than predicated add,  based on GCC -Ofast loop unroll.
    // OLD: ADD  z0.s, p0/M, z0.s, z1.s
    "ADD z0.s, z0.s, z1.s; "

    "ST1W z0.s, p0, [%[vresBase], %[idx], LSL #2]; "

    // Optimization: Simple ADD seems a bit faster than INCW,  based on GCC -Ofast loop unroll.
    // OLD: INCW %[idx]
    "ADD %[idx], %[idx], %[kIdxStep]; "

    "WHILELO p0.s, %[idx], %[arrSize]; "
    "B.FIRST L_loop_begin_%=; "

    "L_loop_end_%=: "
    : // no output
    :
      [arrSize] "r" (arrSize),
      [vresBase] "r" (vresBase),
      [vaBase] "r" (vaBase),
      [vbBase] "r" (vbBase),
      [vaPrefetchBase] "r" (vaPrefetchBase),
      [vbPrefetchBase] "r" (vbPrefetchBase),
      [idx] "r" (idx),
      [kIdxStep] "r" (kIdxStep)
    : "p0", "p1", "z0", "z1"
  );
}
#endif

const size_t kVectorSize  =  1 * 1024 * 1024;
const size_t kRepeatCount    =  100;
const uint64_t kWarmupCount  =  10;

typedef void (*operation_desc_function_t)(std::vector<int32_t>& vres, const std::vector<int32_t>& va, const std::vector<int32_t>& vb);

typedef struct {
  const char* name;
  volatile operation_desc_function_t fn;
  Runtime runtime;
  std::vector<int32_t> result_vector;
} operation_desc_t;

void init_op_descriptors(std::vector<operation_desc_t>& opDescriptors) {
  opDescriptors.push_back({ "c",                  vector_add_i32_c,                   Runtime(kWarmupCount), std::vector<int32_t>(), });
  opDescriptors.push_back({ "c_prefetch",         vector_add_i32_c_prefetch,          Runtime(kWarmupCount), std::vector<int32_t>(), });

  #if defined(__aarch64__)
  opDescriptors.push_back({ "neon",               vector_add_i32_neon,                Runtime(kWarmupCount), std::vector<int32_t>(), });
  opDescriptors.push_back({ "sve",                vector_add_i32_sve,                 Runtime(kWarmupCount), std::vector<int32_t>(), });
  opDescriptors.push_back({ "sve_raw",            vector_add_i32_sve_raw,             Runtime(kWarmupCount), std::vector<int32_t>(), });
  opDescriptors.push_back({ "sve_prefetch",       vector_add_i32_sve_prefetch,        Runtime(kWarmupCount), std::vector<int32_t>(), });
  opDescriptors.push_back({ "sve_prefetch_raw",   vector_add_i32_sve_prefetch_raw,    Runtime(kWarmupCount), std::vector<int32_t>(), });
  #endif
}

std::vector<operation_desc_t> gOpDescriptors;
std::vector< std::shared_ptr<std::vector<operation_desc_t>> > gThreadOpDescriptors;

void worker_thread_main(const int threadIdx, const int iterationCount) {
  std::vector<operation_desc_t>& opDescriptors = *gThreadOpDescriptors[threadIdx];

  int i = 0;
  while(true) {
    printf("[t=%d] i=%d\n", threadIdx, i);

    const std::vector<int32_t> va = makeVector(kVectorSize);
    const std::vector<int32_t> vb = makeVector(kVectorSize);

    // zero out result_vector
    for (auto& opd : opDescriptors) {
      memset(&opd.result_vector[0], 0, kVectorSize);
    }

    // run through all operations
    for (auto& opd : opDescriptors) {

      opd.runtime.begin();

      // run the test function kRepeatCount times
      for (int j = 0; gKeepRunning && j < kRepeatCount; ++j) {

        opd.fn(opd.result_vector, va, vb);
        opd.runtime.record(kVectorSize);

        // check result against C
        if (!compareVectors(opd.result_vector, opDescriptors[0].result_vector)) {
          printf("[t=%d] %s fail on i=%d!\n", threadIdx, opd.name, i);
          vector_print_i32_c(threadIdx, opd.name, opd.result_vector);
          vector_print_i32_c(threadIdx, opDescriptors[0].name, opDescriptors[0].result_vector);

          // stop thread on error
          gHasErrors = true;
          return;
        }
      }
    }

    if (iterationCount > 0 and i >= iterationCount) {
      printf("[t=%d] stop on limit\n", threadIdx);
      return;
    }

    if (!gKeepRunning) {
      printf("[t=%d] stop on signal\n", threadIdx);
      return;
    }

    // next iteration
    ++i;
  } // end loop
}

int main(int argc, char** arvg) {
  signal(SIGINT, handle_process_exit_signal);

  srand(getpid() * time(NULL));

  init_op_descriptors(gOpDescriptors);

  int iterationCount = 0;
  if (argc >= 2) {
    iterationCount = std::atoi(arvg[1]);
  }

  const long int cpuCount = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpuCount < 1) {
    printf("failed to get cpuCount %ld, errno %d\n", cpuCount, errno);
    return 1;
  }

  for (int threadIdx = 0; threadIdx < cpuCount; ++threadIdx) {
    std::shared_ptr<std::vector<operation_desc_t>> threadOpDescriptors = std::shared_ptr<std::vector<operation_desc_t>>(new std::vector<operation_desc_t>());

    init_op_descriptors(*threadOpDescriptors);

    // warm up result vector sizes
    for (auto& opd : *threadOpDescriptors) {
      opd.result_vector.resize(kVectorSize);
    }

    gThreadOpDescriptors.push_back(threadOpDescriptors);
  }

  printf("creating %ld threads\n", cpuCount);
  std::vector<std::shared_ptr<std::thread>> threads;
  for (int threadIdx = 0; threadIdx < cpuCount; ++threadIdx) {
    std::shared_ptr<std::thread> thr(new std::thread(&worker_thread_main, threadIdx, iterationCount));
    threads.push_back(thr);
  }

  for (auto thr : threads) {
    thr->join();
  }

  for (int opdIndex = 0; opdIndex < gOpDescriptors.size(); ++opdIndex) {
    for (int threadIdx = 0; threadIdx < cpuCount; ++threadIdx) {
      gOpDescriptors[opdIndex].runtime.add((*gThreadOpDescriptors[threadIdx])[opdIndex].runtime);
    }
  }

  printf("Run times per element:\n");
  for (int opdIndex = 0; opdIndex < gOpDescriptors.size(); ++opdIndex) {
    gOpDescriptors[opdIndex].runtime.println(gOpDescriptors[opdIndex].name);
  }

  int exitCode = 0;
  if (gHasErrors) {
    exitCode = 1;
  }

  printf("exitCode=%d\n", exitCode);
  return exitCode;
}

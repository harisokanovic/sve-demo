// Runtime.cpp

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#include <limits>
#include <stdexcept>

#include <Runtime.h>

static uint64_t getMonoTimeNs() {
  struct timespec ts = {0};
  const int status = clock_gettime(CLOCK_MONOTONIC, &ts);
  if (status != 0) {
    throw std::runtime_error("clock_gettime() failed");
  }

  return (ts.tv_sec * 1000000000) + ts.tv_nsec;
}

Runtime::Runtime(const uint64_t warmupCount)
  : mWarmupCount(warmupCount),
    mWarmupCountRemaining(0),

    mBeginTime(0),

    mCount(0),
    mSum(0)
{ }

Runtime::Runtime(const Runtime& other)
  : mWarmupCount(other.mWarmupCount),
    mWarmupCountRemaining(other.mWarmupCountRemaining),

    mBeginTime(other.mBeginTime),

    mCount(other.mCount),
    mSum(other.mSum)
{ }

Runtime::~Runtime()
{ }

void Runtime::begin() {
  mWarmupCountRemaining = mWarmupCount;
  mBeginTime = getMonoTimeNs();
}

void Runtime::record(const uint64_t elementCount) {
  const uint64_t currentTime = getMonoTimeNs();
  const uint64_t deltaTime = currentTime - mBeginTime;
  mBeginTime = currentTime; // for next iteration

  if (deltaTime < 1) {
    // should never happen, avoid divide by zero
    throw std::runtime_error("zero runtime");
  }

  if (mWarmupCountRemaining == 0) {
    mCount += elementCount;
    mSum += deltaTime;
  } else {
    --mWarmupCountRemaining;
  }
}

void Runtime::add(const Runtime& other) {
  mCount += other.mCount;
  mSum += other.mSum;
}

void Runtime::println(const char* const name) {
  const double am = mSum / std::max(mCount, 1.0d);
  printf("%s = {count=%.0f, am=%.2f ns}\n",
    name, mCount, am);
}

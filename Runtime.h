// Runtime.h

#ifndef RUN_TIME_CLASS_H
#define RUN_TIME_CLASS_H

#include <stdint.h>

class Runtime {
public:
  Runtime(const uint64_t warmupCount);
  Runtime(const Runtime& other);
  ~Runtime();

  void begin();
  void record(const uint64_t elementCount);

  void add(const Runtime& other);

  void println(const char* const name);

private:
  const uint64_t mWarmupCount;
  uint64_t mWarmupCountRemaining;

  uint64_t mBeginTime;

  double mCount;
  double mSum;
};

#endif

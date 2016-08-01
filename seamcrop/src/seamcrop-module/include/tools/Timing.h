#ifndef TOOLS_TIMING_H
#define TOOLS_TIMING_H

#include "types/MocaTypes.h"
#include <vector>

#if defined WIN32

// Windows
#include <Windows.h>

#elif defined DARWIN

// Mac OS X
#include <mach/mach_time.h>

#else

// Linux
#include <time.h>
#include <unistd.h>
#if _POSIX_TIMERS <= 0
#error "The necessary timing functions aren't available on this system."
#endif // _POSIX_TIMERS
#ifndef _POSIX_MONOTONIC_CLOCK
#error "The necessary timing functions aren't available on this system."
#endif // _POSIX_MONOTONIC_CLOCK

#endif


class Timing
{
 public:
  static void start(indexType timer = 0);
  static double stop(indexType timer = 0);
  static double stopAndPrint(indexType timer = 0);
  static void reset(indexType timer = 0);
  static double sum(indexType timer = 0);
  static double avg(indexType timer = 0);
  static uint32 count(indexType timer = 0);

 private:
  Timing(sizeType noOfTimers);
  ~Timing();
  static Timing& instance();

  static Timing* _instance;

  #if defined WIN32
  LARGE_INTEGER freq;
  std::vector<LARGE_INTEGER> timers;
  #elif defined DARWIN
  mach_timebase_info_data_t timeinfo;
  std::vector<uint64> timers;
  #else
  std::vector<timespec> timers;
  #endif
  std::vector<uint32> counters;
  std::vector<double> sums;
};

#endif

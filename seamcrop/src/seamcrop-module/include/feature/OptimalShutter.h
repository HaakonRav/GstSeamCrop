#ifndef FEATURE_OPTIMAL_SHUTTER_H
#define FEATURE_OPTIMAL_SHUTTER_H

#include "types/MocaTypes.h"
#include <vector>
class Histogram;
class Exposure;


class OptimalShutter
{
  static const double HIGH_SHUT = 4096;
  typedef std::vector<double> DVec; // keep our signatures clean
 public:
  static DVec findBestShutters(Histogram const& hist, uint32 numExps, double maxCoverage = 10.0, double maxShutterSum = 400000.0);

  static std::vector<Exposure> selectExposures(std::vector<Exposure> const& expSet, DVec const& shutters);

  static DVec createWeightMask(Histogram const& hist);
  static DVec createCoverage(Histogram const& hist, DVec const& shutters, DVec const& weighting);
  static double estimateCoverage(Histogram const& hist, DVec const& coverage);

  static DVec findBestShutters(Histogram const& hist, DVec const& weighting, uint32 numExps, double maxCoverage = 10.0, double maxShutterSum = 400000.0);

  static double findNextShutter(Histogram const& hist, DVec const& weighting, DVec const& coverage, bool maxShut);
  
  /// computes equidistant shutter values that cover the entire dynamic range
  static DVec findEquiShutters(uint32 numExps, Histogram const& hist);
};


#endif

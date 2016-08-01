#ifndef FEATURE_HISTOGRAM_REG_H
#define FEATURE_HISTOGRAM_REG_H

#include "types/MocaTypes.h"
#include "tools/KalmanFilter.h"
#include "types/Image8U.h"
#include "types/Vector.h"
#include <vector>
class Exposure;

class HistogramReg
{
public:
  HistogramReg(int32 maxOffset, double maxConf);

  /// offset (in): a priori estimated offset
  /// offset (out): offset determined by histogram registration
  /// confidence (out): 2D vector of the confidence in the returned offset
  void computeShift(Exposure const& exp1, Exposure const& exp2, VectorI& offset, Vector& confidence);
  void initFilter();
  VectorI filterOffset(VectorI const& offset, Vector const& confidence);

 protected:
  void computeMeans(Exposure const& exp1, Exposure const& exp2, uint8& mean1, uint8& mean2);
  int32 findMean(VectorI const& hist, int32 percentage);

  KalmanFilter kf;

  int32 maxOffset;
  double maxConf;
  uint8 noiseThresh; // noise threshold of the MTB
  double obsCovFactor; // scaling factor between observation covariance and confidence
  uint8 badMean; // [0..badMean) and (255-badMean..255] are considered invalid means for the MTB
  uint8 nextMeanStep; // distance to 50% image brightness in case of a bad mean
  bool startWithRows;
  uint32 nccIters; // number of iterations of the entire algorithm
  double initCov, procCov;
};

#endif
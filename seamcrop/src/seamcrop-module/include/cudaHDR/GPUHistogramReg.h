#ifndef FEATURE_GPU_HISTOGRAM_REG_H
#define FEATURE_GPU_HISTOGRAM_REG_H

#include "feature/HistogramReg.h"
#include "types/MocaTypes.h"
#include "tools/KalmanFilter.h"
#include "types/Image8U.h"
#include "types/Vector.h"
#include "cudaHDR/CudaExposureHandle.h"
#include "cudaHDR/CudaVectorHandle.h"
#include "cudaHDR/CudaWrapper.h"

#include <vector>
class Exposure;

class GPUHistogramReg : public HistogramReg
{
 public:
  GPUHistogramReg(int32 maxOffset, double maxConf) : HistogramReg(maxOffset, maxConf) {};
  void computeShift(CudaExposureHandle const& exp1, 
	  CudaExposureHandle const& exp2, 
	  VectorI& offset, 
	  Vector& confidence);

 protected:
  int findMean(int* histo, int numpixels, int percentage);
  void computeMeans(CudaExposureHandle const& exp1, 
		CudaExposureHandle const& exp2, 
		uint8& mean1, 
		uint8& mean2);
};

#endif

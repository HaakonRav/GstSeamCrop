#ifndef FEATURE_GPU_TONEMAPPING_H
#define FEATURE_GPU_TONEMAPPING_H

#include "types/Image8U.h"
#include "feature/Histogram.h"
#include "filter/ToneMapping.h"

#include "cudaHDR/CudaImage32FHandle.h"
#include "cudaHDR/CudaImage8UHandle.h"


class GPUToneMapping : public ToneMapping
{
 public:
  static Histogram histNorm(CudaImage32FHandle const& image, CudaImage8UHandle& result);
};


#endif

#ifndef FILTER_FREQUENCYTUNEDSALIENCY_H
#define FILTER_FREQUENCYTUNEDSALIENCY_H

#include "types/Image8U.h"
#include "types/Image32F.h"
#include "types/MocaTypes.h"

#include <vector>

class FrequencyTunedSaliency
{
 public:
  
  static void calculate(Image8U const& image, Image32F& result, bool const& normflag=true);
  
private:
  static void RGB2LAB(Image8U const& image, std::vector<double>& lvec, std::vector<double>& avec, std::vector<double>& bvec);
  
  static void GaussianSmooth(std::vector<double> const& inputImg, int const& width, int const& height, std::vector<double> const& kernel, std::vector<double>& smoothImg);
  
  static void normalize(Image32F& image, uint32 const& normrange=255);
  
};

#endif

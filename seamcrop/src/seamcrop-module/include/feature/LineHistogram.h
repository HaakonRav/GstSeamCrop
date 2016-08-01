#ifndef LINE_HISTOGRAM_H
#define LINE_HISTOGRAM_H

#include "types/MocaTypes.h"
#include "tools/KalmanFilter.h"
#include "types/Image8U.h"
#include "types/Vector.h"
#include <vector>
class Exposure;

class LineHist
  {
  public:
    LineHist();
    LineHist(Rect roi);
    void create(Image8U const& image, uint8 mean, uint8 noiseThresh, bool rows);
    int32 corrOffset(LineHist const& other, int32 guess, int32 maxOffset);
    
    std::vector<uint32> white, black;
    Rect roi;
  };

#endif
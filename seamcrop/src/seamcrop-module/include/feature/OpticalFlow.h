#ifndef TOOLS_OPTICALFLOW_H
#define TOOLS_OPTICALFLOW_H

#include "types/MocaTypes.h"
#include "types/Image8U.h"
#include "types/Vector.h"

#include <vector>
#include <utility>

class OpticalFlow 
{
  public:
    typedef std::vector<std::pair<Vector, Vector> > features;
    typedef boost::shared_ptr<features> featuresPointer;
    
    static featuresPointer calcOpticalFlow(Image8U const& firstImage,Image8U const& secondImage,int maxFeatures, Vector* srcFeatures=NULL); 
  
  private:
    static void getGoodFeatures(IplImage* firstImage, CvSize& frameSize, int& maxFeatures, CvPoint2D32f* srcFeatures);
};

#endif

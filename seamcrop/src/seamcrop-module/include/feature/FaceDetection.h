#ifndef TOOLS_FACEDETECTION_H
#define TOOLS_FACEDETECTION_H

#include "types/MocaTypes.h"
#include "types/Image8U.h"
#include <vector>

enum cascadeType
  {
    FRONTAL_FACE,
    PROFILE_FACE,
    FULL_BODY    
  };

class FaceDetection 
{
  public:
    FaceDetection(cascadeType cascade);
    
    void detect(Image8U const& srcImage, std::vector<Rect>& objects);
    
  private:
    const char* cascadeFile;
    CvHaarClassifierCascade* cascade;
};

#endif
 

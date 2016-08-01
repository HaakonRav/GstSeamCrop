#pragma once
#include <opencv2/opencv.hpp>

#include "types/Image8U.h"
#include "types/Image32F.h"

class DisparityMapBuilder
{
private:
  void split_vertical(IplImage* img, IplImage* left_eye_view, IplImage* right_eye_view);

public:
  void buildDisparityMapBM(Image8U const& img, Image8U& dest);
  void buildDisparityMapSGBM(Image8U const& img, Image8U& dest);
};

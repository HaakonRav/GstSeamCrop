#ifndef FILTER_CVTCOLORSPACE_H
#define FILTER_CVTCOLORSPACE_H

#include <map>
#include "types/MocaTypes.h"
class ImageBase;
class Image32F;
class Image8U;


enum ColorSpace
  {
    COLOR_RGB=1,
    COLOR_Y=2,
    COLOR_BGR=4,
    COLOR_XYZ=8,
    COLOR_YCrCb=16,
    COLOR_HSV=32,
    COLOR_HLS=64,
    COLOR_Lab=128,
    COLOR_Luv=256,
    COLOR_BayerBG=512,
    COLOR_BayerGB=1024,
    COLOR_BayerRG=2048,
    COLOR_BayerGR=4096,
    COLOR_Luv8=8192,
    COLOR_Yxy=16384
  };


class CvtColorSpace
{
 public:
  /// Converts an Image into an other Color Space
  static void convert(Image8U const& image, Image8U& result, ColorSpace srcMode, ColorSpace destMode);

  /// Converts Image32F BGR to Yxy.
  static void cvtBGR_Yxy(Image32F const& image, Image32F& result);
  /// Converts Image32F Yxy to BGR
  static void cvtYxy_BGR(Image32F const& image, Image32F& result);

 private:
  /// Converts YCrCb444 to YCrCb8. Only needed by convertColorSpace(...);
  static void cvtLuv_Luv8(Image8U const& image, Image8U& result);
  /// Converts YCrCb8 to YCrCb444. Only needed by convertColorSpace(...);
  static void cvtLuv8_Luv(Image8U const& image, Image8U& result);
  /// Converts BGR to Yxy.
  static void cvtBGR_Yxy(Image8U const& image, Image8U& result);
  /// Converts Yxy to BGR
  static void cvtYxy_BGR(Image8U const& image, Image8U& result);
  /// Ensures that the border of the image isn't just black
  static void fixBayerResult(Image8U& result);

  /// initializes spaceMap. Allowed conversions must be defined here.
  static void initColorSpaceMap();
  /// stores correspondences between MoCA and CV color spaces
  static std::map<uint64, int32> spaceMap;
};

#endif 

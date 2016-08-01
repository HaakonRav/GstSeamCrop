#ifndef FEATURE_FEATURE_H
#define FEATURE_FEATURE_H

#include "types/MocaTypes.h"
#include "types/Vector.h"

#include <vector>

class ImageBase;
class Image8U;
class Image16S;
class Image32F;
class CameraImage;
class Color;
class Rect;


class Feature
{
 public:
  /// Computes the center of gravity of a colored area.
  static void computeCoG(ImageBase const& image, double& x, double& y, Rect const area, int32 compColor);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void computeHist(ImageBase const& image, VectorI& hist, double normFactor = -1);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  // the parameter features was of type ImageFeatureSet in VideoAOI. Maybe this should be changed back as soon as the required classes are ported to MoCA.
  static void goodFeatures(ImageBase const& image, std::vector<Vector>& features, double qualityLevel, double minDist, int maxFeatures);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void minMaxLoc(ImageBase const& image, double& minVal, double& maxVal, VectorI& minLoc, VectorI& maxLoc);
  /// Converts a Hue/Saturation/Value color value to RGB.
  static void convertHSVRGB(double h, double s, double v, Color& color);

  /// Returns a color value "between the pixels" by bilinear interpolating the surrounding pixel values.
  /// Only considers the first channel.
  static void pixelBilerp(Image8U const& image, double& value, double x, double y);
  static void pixelBilerp(CameraImage const& image, double& value, double x, double y);
  static void pixelBilerp(Image16S const& image, double& value, double x, double y);
  static void pixelBilerp(Image32F const& image, double& value, double x, double y);
  /// Returns a smoothed pixel by applying a 3 by 3 gaussian mask to the pixel.
  static void pixel8UGauss(Image8U const& image, double& value, unsigned int x, unsigned int y);
  /// Computes a contrast-based saliency map for an image
  static void saliencyMap(Image8U const& image, Image8U& saliency, int radius, double sigma);
};

#endif


#ifndef FILTER_TONE_MAPPING_H
#define FILTER_TONE_MAPPING_H

#include "feature/Histogram.h"
#include "types/MocaTypes.h"
#include "tools/Maths.h"
#include <limits>

class Image32F;
class Image8U;

/// All methods convert a 32 bit floating point image into an 8 bit unsigned image performing tone mapping.
class ToneMapping
{
 public:
  /// Uses a simple logarithm and scaling. Returns average brightness of the 32F image as a dirty hack...
  /// Both images are assumed to be B/W or in Yxy color space.
  static float simpleLog(Image32F const& image, Image8U& result);

  /// Uses Drago's logarithmic mapping. Returns average brightness. Assumes B/W or Yxy
  static float dragoLog(Image32F const& image, Image8U& result);

  /// Uses Ward's contrast-based scale factor. Returns average brightness. Scales additionally.
  /// Input Yxy. Output BGR.
  static float wardContrast(Image32F const& image, Image8U& result, float scale);

  /// Uses Ward histogram normalization for range compression.
  /// If scale>0, scales brightness and converts to BGR. Yxy otherwise.
  static Histogram histNorm(Image32F const& image, Image8U& result, float scale=-1);

  /// Reinhard's photographic tone mapping operator. Scales additionally. Input Yxy. Output BGR.
  static float photographic(Image32F const& hdrImage, Image8U& result, float scale);


  // ========== histogram normalization ==========
  ///Trims a histogram by capping the peak
  static double trimHisto(Histogram& hist);
  ///Normalizes and cumulates a histogram
  static void cumulateAndNormalizeHisto(Histogram& hist);
  //Performs tonemapping with a given histogram
  static void mapImageHist(Image32F const& image, Image32F& result, Histogram const& hist);
  ///Computes a logarithmic histogram
  static void logHisto(Image32F const& image, Histogram& hist, float inc = 1.0f);
  
  // ========== photographic operator ==========
  /// Pre-scaling of the Y channel before the real tone mapping is done. key is the scaling parameter.
  static void scaleToMidtone(Image32F& image, float key);
  /// Builds a gaussian pyramid of the pre-scaled HDR image. range is the number of levels.
  static void buildPyramid(Image32F const& image, std::vector<Image32F>& result, uint32 range);
  /// Performs the actual tone mapping. Result is still Yxy.
  static void mapImagePhoto(std::vector<Image32F> const& pyramid, Image32F& image, float key);

  // ========== misc functions ==========
  /// Returns the min/max pixel value of the image's first channel and the average of the log image
  /// pixel values will automatically be raised to >= minVal
  static void minMaxAvg(Image32F const& image, float& min, float& max, float& avg, float minVal=-std::numeric_limits<float>::infinity());
  /// Returns the luminance of the pixel at the given position. It's either the first channel (b/w or Yxy) or a mix of all three (BGR).
  static float luminance(Image32F const& image, uint32 x, uint32 y, bool isBGR=false);
  /// Normalizes all channels of the image to [0..range]. Returns average from minMaxAvg.
  static float normalize(Image32F& image, float range);
  /// Applies pixval^(1/gamma) to all channels of the image (RGB and BGR)
  static void applyGamma(Image32F& image, float gamma);
};

#endif

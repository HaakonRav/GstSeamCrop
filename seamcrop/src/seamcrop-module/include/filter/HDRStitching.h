#ifndef FILTER_HDRSTITCHING_H
#define FILTER_HDRSTITCHING_H

#include "types/MocaTypes.h"
#include "types/Image8U.h"
#include "types/Image32F.h"
#include <boost/shared_ptr.hpp>
#include <vector>

class Exposure;


class HDRStitching
{
 public:
  /// uses the brightest non-saturated pixel to compute the radiance
  static void linear(std::vector<Exposure> const& exposures, Image32F& result, uint32 brightest);
  /// uses a weighted average of all pixels to compute the radiance
  static void weighted(std::vector<Exposure> const& exposures, Image32F& result);
  /// same as weighted() but with added ghost removal  
  static void weightedNoGhosts(std::vector<Exposure> const& exposures, Image32F& result, uint32 brightest, float thresh);
  /// returns the binary image containing the regions detected as ghost regions. The return value is only valid after a call to weightedNoGhosts()
  static boost::shared_ptr<Image8U> getGhostSegments();
  
  static int32 const reductionFactor = 10; // determines how much smaller the ghost segment images are (each dimension gets divided by this value)

 private:
  /// calculates which regions of varImage are ghost regions using threshold for creating the underlying binary image
  static void findGhostSegments(Image32F const& varImage, Image8U& result, double threshold);
  /// removes all continguous regions (of color oldRegionCol) from image that cover an area of less than segmentLimit (removed regions are in color nonRegionCol)
  static void removeLowAreaSegments(Image8U& image, uint8 oldRegionCol, uint8 newRegionCol, uint8 nonRegionCol, double segmentLimit);
  /// determines which HDR-value is best suited for choosing an exposure to create an LDR-frame of roi (only ghost regions are considered)
  static float findBestValue(Image32F const& image, Image8U const& ghostSegments, Rect const& roi);
  /// creates and "LDR"-frame of roi using only as few different exposures as possible. shutter determines how exposures are ranked when deciding between them
  static void findBestExposure(std::vector<Exposure> const& exposures, Image32F& result, float shutter, Rect const& roi);
  /// copies the singleExposure into the corresponing region in result (only for pixel which belong to ghost regions)
  static void combine(Image32F& result, Image32F const& singleExposure, Image32F const& var, Image8U const& ghostSegments, Rect const& roi);
  /// smooths those pixel that lie at the edge of a ghost region
  static void smoothEdges(Image32F& result, Image8U const& ghostSegments, int32 roiSize);

  static boost::shared_ptr<Image8U> ghostSegments;
  
  static double const minRelativeSegmentSize;  // is relative to the reduced image size (thus compared to the original image valid areas
                                               // are even smaller relativly speaking)
  static double const histogramStepSize;       // size of each bin of the histogram used for findBestValue()
  static int32 const minNumBins;               // minimum number of bins used for the histogram
  static double const relativeNumOutliers;     // relative number values regarded as outliers in findBestValue()
  static double const smoothingStrength ;      // value passed to Filter::smooth() for smoothing the edges
};


#endif

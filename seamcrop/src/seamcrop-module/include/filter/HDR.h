#ifndef FILTER_HDR_H
#define FILTER_HDR_H

#include "types/MocaTypes.h"
#include "types/Exposure.h"
#include "types/Vector.h"
#include "types/Rect.h"
#include <boost/shared_ptr.hpp>
#include <vector>
#include <queue>

class CameraReader;
class DC1394Reader_Triggered;
class Histogram;
class Image8U;
class Image32F;


class HDR
{
 public:
  HDR(boost::shared_ptr<CameraReader> reader, uint32 darkest, uint32 brightest, double minInvalidPerLine);

  void captureFullHDR(std::vector<Exposure>& exposures);
  /// Uses the shutter speed and roi from the first exposure in the set to start HDR capturing
  /// Drops all other exposures.
  virtual void captureHDR(std::vector<Exposure>& exposures);

  void colorThresholdImage(Image8U const& image, Image8U& imgCol, bool linewise);
  void virtualExposure(Image32F const& hdrImage, Image8U& ldrImage, uint32 shutter); // possible static

 protected:
  /// puts detected regions of re-exposure into the exposure queue
  /// sets the next shutter speed according to currExposure and direction
  void enqueueExposures(Exposure const& currExposure, Exposure::CheckedInvalids direction, std::vector<Rect> const& rois, std::queue<Exposure>& exposures, uint32 parent);

  /// analyzes the image for under- or overexposed pixels (according to direction)
  /// puts the found regions into the proper vectors
  void nextExposureRoi(Image8U const& image, Exposure::CheckedInvalids direction, std::vector<Rect>& darkRois, std::vector<Rect>& brightRois, uint32& tooDark, uint32& tooBright);
  void nextExposureRoi(Image8U const& image, Exposure::CheckedInvalids direction, std::vector<Rect>& darkRois, std::vector<Rect>& brightRois);
  /// convenience function if direction != TOO_BOTH
  void nextExposureRoi(Image8U const& image, Exposure::CheckedInvalids direction, std::vector<Rect>& rois);
  void nextExposureRoi(Image8U const& image, Exposure::CheckedInvalids direction, std::vector<Rect>& rois, uint32& tooDark, uint32& tooBright);
 
  /// counts invalid pixels in an image. returns total numbers and invalids per line
  void countInvalidPixels(Image8U const& image, std::vector<uint32>& tooDarkPerLine, std::vector<uint32>& tooBrightPerLine, uint32& tooDark, uint32& tooBright);
  void countInvalidPixels(Image8U const& image, uint32& tooDark, uint32& tooBright);
  void countInvalidPixels(Image8U const& image, std::vector<uint32>& tooDarkPerLine, std::vector<uint32>& tooBrightPerLine);

  /// factorizes the total exposure into suitable values for shutter and gain
  /// returns false if total exposure is out of bounds
  void exposureToShutGain(float totalExp, uint32& shutter, float& gain, bool shutFirst=true);

  boost::shared_ptr<CameraReader> reader;
  uint32 darkest, brightest;
  sizeType unitSizeX, unitSizeY; // smallest units of image size in the camera
  uint32 unitPosX, unitPosY; // smallest units of image positions
  Rect roi; // maximum image dimensions
  double minInvalidPerLine; // minimum invalid pixels per line to do a re-exposure
  uint32 minShutter, maxShutter; // smallest and largest shutter values of the camera
  float minGainFact, maxGainFact; // boundaries of the gain amplification. These are _not_ camera values
};


#endif


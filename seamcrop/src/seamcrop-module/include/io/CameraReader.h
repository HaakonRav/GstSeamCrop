#ifndef IO_CAMERAREADER_H
#define IO_CAMERAREADER_H

#include "io/Reader.h"
#include "io/CameraFeature.h"
#include "types/CameraImage.h"
#include <boost/shared_ptr.hpp>


/**
   Base class for readers that capture live images from a camera.
   This class is mostly used for platform independence.
   The two major implementations of CameraReader are the linux camera reader using the libdc1394 library
   and its windows counterpart using FireGrab.
   This class implements the manipulation of camera features.
**/
class CameraReader : public Reader
{
public:
  enum BusSpeed { SPEED_400, SPEED_800 };
  enum ColorMode { MODE_MONO8, MODE_RAW8 }; // MONO8 is B/W, RAW8 is color (bayer pattern)

  /// Constructor, duh!
  CameraReader(ColorMode colorMode);
  /// Destructor.
  virtual ~CameraReader() {}
  /// Queries the camera's range of values for the given feature.
  void getFeatureBoundaries(CameraFeature::Type feature, uint32 &min, uint32 &max);
  /// Queries the currently set value of a feature from the camera.
  uint32 getFeatureValue(CameraFeature::Type feature);
  /// Sets the feature to the given value inside the camera.
  void setFeatureValue(CameraFeature::Type feature, uint32 newVal);
  /// Returns the CameraFeature of the given position in the list. The feature pointer can be used to modify the feature.
  CameraFeaturePtr getFeature(indexType index);
  /// Returns the CameraFeature of the given type. The feature pointer can be used to modify the feature.
  CameraFeaturePtr getFeature(CameraFeature::Type feature);
  /// Returns the color mode.
  ColorMode getColorMode();

  /// Returns maximum settable image size for this camera.
  virtual void getMaxImageSize(sizeType& width, sizeType& height) = 0;
  /// Returns the smallest increment of image size for this camera. These are usually values like 2 or 4.
  virtual void getUnitSize(sizeType& unitSizeX, sizeType& unitSizeY) = 0;
  /// Returns the smallest increment of image position for this camera. These are usually values like 2 or 4.
  virtual void getUnitPos(uint32& unitPosX, uint32& unitPosY) = 0;

  /// Triggers the camera using absolute times as shutter and gain value
  virtual void captureImage(Rect roi, uint32 shutter, float gain=-10034) = 0;
  
  virtual boost::shared_ptr<CameraImage> getImagePtr() = 0;
  virtual void returnImagePtr(boost::shared_ptr<CameraImage> image) = 0;

protected:
  std::vector<CameraFeaturePtr> features; /// List of all possible features of the camera (both existing and non-existing).
  std::map<CameraFeature::Type, int> featIndices; /// Maps feature types to the corresponding index in the feature array
  
  ColorMode colorMode;
};

#endif


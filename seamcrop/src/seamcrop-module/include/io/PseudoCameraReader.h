#ifndef IO_PSEUDOCAMERAREADER_H
#define IO_PSEUDOCAMERAREADER_H


#include "io/CameraReader.h"


/**
   Reader that simulates a real camera and live capturing capabilities.
   It will read a set of images taken at different brightness levels from disk and store them internally.
   Each call to captureImage() and getImage() will then copy the specified area from a saved image and return it.
   This reader is mostly used to test algorithms that were implemented by capturing live images.
   A real camera reader can easily be replaced with a pseudo reader which makes the results reproducible.
**/
class PseudoCameraReader : public CameraReader
{
 public:
  /// Constructor. Loads the first image of the sequence to see if it exists and determines its size.
  PseudoCameraReader(std::string fileName);
  virtual ~PseudoCameraReader();
 
  void start(); /// Does nothing.
  void stop(); /// Does nothing.
  
  /// just "implemented" because to abide by the interface
  void captureImage(Rect, float, float) {};
  /// just "implemented" because to abide by the interface
  boost::shared_ptr<CameraImage> getImagePtr();
  /// just "implemented" because to abide by the interface
  void returnImagePtr(boost::shared_ptr<CameraImage>) {};

  /// Calculates an image index from the given shutter value and copies the given image area from the image with the computed index into member capturedImage.
  void captureImage(Rect roi, uint32 shutter, float gain);
  /// Returns a copy of the image saved in member capturedImage.
  void getImage(Image8U& img);

  /// Returns the width of the saved images.
  sizeType getImageWidth();
  /// Returns the height of the saved images.
  sizeType getImageHeight();
  /// Returns Rect(0, 0, width(), height())
  Rect getImageDimension();
  /// Retuns the width and height of the saved images (only implemented to abide by the interface).
  void getMaxImageSize(sizeType& imageWidth, sizeType& imageHeight);
  /// Returns the number of channels.
  int32 getImageChannels();
  
  /// Returns (4, 2). These values were set to emulate the Marlin camera.
  void getUnitSize(sizeType& unitSizeX, sizeType& unitSizeY);
  /// Returns (2, 2). These values were set to emulate the Marlin camera.
  void getUnitPos(uint32& unitPosX, uint32& unitPosY);

 protected:
  bool loadImages;
  /// Base file name of the saved images. Does not contain index or extension.
  std::string fileName;
  sizeType unitSizeX, unitSizeY;
  uint32 unitPosX, unitPosY;
  sizeType imageWidth, imageHeight;
  int32 imageChannels;
  /// Stores the image in memory that is going to be retrieved next. This member is filled in captureImage().
  boost::shared_ptr<Image8U> capturedImage;
};

#endif


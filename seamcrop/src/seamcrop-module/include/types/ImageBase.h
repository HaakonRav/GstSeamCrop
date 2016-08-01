#ifndef TYPES_IMAGE_BASE_H
#define TYPES_IMAGE_BASE_H

#include "types/MocaTypes.h"
#include "types/Rect.h"

#include <iostream>
#include <opencv/cv.h>

class ImageBase
{
 public:
  // This copy constructor is only here for MocaException. Do NOT use elsewhere!
  ImageBase(ImageBase const& other);
  
  virtual ~ImageBase();

  virtual ImageBase& operator=(ImageBase const& other);
  
  void copyFrom(IplImage const* other);
  void copyFrom(ImageBase const& other);
  //void resize(sizeType width, sizeType height);
  //void realloc const ImageSize& size, const ImageSize& tileSize, BorderMode borderMode);
  
  Rect getRoi() const;
  void setRoi(Rect const& newRoi);
  void resetRoi();

  sizeType width() const;
  sizeType height() const;
  int32 channels() const;
  int32 depth() const;
  int32 widthStep() const;
  
  uint64 getTimestamp() const;
  void setTimestamp(uint64 newTimestamp);
  
  bool matchingParams(IplImage const* other) const;
  bool matchingParams(ImageBase const& other) const;

  uint8* ptr();
  uint8 const* ptr() const;
  
  IplImage* image; // Use this pointer for calling OpenCV functions

 protected:
  ImageBase(sizeType width, sizeType height, int32 channels, int32 depth);
  ImageBase(IplImage* image);

  // only used for convenience in the sub-classes
  inline uint8& pixel(indexType x, indexType y, indexType band=0)
  {
    // enable in case of a subtle segfault for easier debugging
    // assert(x >= 0 && x < width() && y >= 0 && y < height());
    return *(uint8*)(image->imageData + image->widthStep*y + bpp*(image->nChannels*x + band));
  }

  inline const uint8& pixel(indexType x, indexType y, indexType band=0) const
  {
    // assert(x >= 0 && x < width() && y >= 0 && y < height());
    return *(uint8*)(image->imageData + image->widthStep*y + bpp*(image->nChannels*x + band));
  }

  uint8 bpp; // bytes per pixel
  uint64 timestamp;
};


// "toString()" method for debugging
std::ostream& operator<<(std::ostream& stream, IplImage const& image);
std::ostream& operator<<(std::ostream& stream, ImageBase const& image);


#endif

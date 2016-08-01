#ifndef TYPES_IMAGE8U_H
#define TYPES_IMAGE8U_H

#include "types/ImageBase.h"


class Image8U : public ImageBase
{
 public:
  Image8U();
  Image8U(sizeType width, sizeType height, int32 channels = 1);
  Image8U(IplImage* image);
  Image8U(Image8U const& other);
  ~Image8U();

  Image8U& operator=(Image8U const& other);
  
  inline uint8& operator()(indexType x, indexType y, indexType band=0)
    { return pixel(x, y, band); }

  inline const uint8& operator()(indexType x, indexType y, indexType band=0) const
    { return pixel(x, y, band); }
};


#endif

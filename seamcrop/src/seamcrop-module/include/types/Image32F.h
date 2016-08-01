#ifndef TYPES_IMAGE32F_H
#define TYPES_IMAGE32F_H

#include "types/ImageBase.h"


class Image32F : public ImageBase
{
 public:
  Image32F();
  Image32F(sizeType width, sizeType height, int32 channels = 1);
  Image32F(IplImage* image);
  Image32F(Image32F const& other);
  ~Image32F();

  Image32F& operator=(Image32F const& other);

  inline float& operator()(indexType x, indexType y, indexType band=0)
    { return (float&) pixel(x, y, band); }
  inline const float& operator()(indexType x, indexType y, indexType band=0) const
    { return (float&) pixel(x, y, band); }
};


#endif

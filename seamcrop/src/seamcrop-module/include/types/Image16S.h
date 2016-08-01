#ifndef TYPES_IMAGE16S_H
#define TYPES_IMAGE16S_H

#include "types/ImageBase.h"


class Image16S : public ImageBase
{
 public:
  Image16S();
  Image16S(sizeType width, sizeType height, int32 channels = 1);
  Image16S(IplImage* image);
  Image16S(Image16S const& other);
  ~Image16S();

  Image16S& operator=(Image16S const& other);

  inline int16& operator()(indexType x, indexType y, indexType band=0)
    { return (int16&) pixel(x, y, band); }
  inline const int16& operator()(indexType x, indexType y, indexType band=0) const
    { return (int16&) pixel(x, y, band); }
};


#endif

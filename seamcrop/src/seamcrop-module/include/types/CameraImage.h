#ifndef TYPES_CAMERAIMAGE_H
#define TYPES_CAMERAIMAGE_H

#include "types/MocaTypes.h"
#include "types/ImageBase.h"

#ifdef HAVE_LIBDC1394
#include <dc1394/control.h>
#endif


class CameraImage : public ImageBase
{
#ifdef HAVE_LIBDC1394

 public:
  CameraImage(dc1394video_frame_t* frame);
   
 private:
  dc1394video_frame_t* frame;
  
  friend class DC1394Reader;
  friend class DC1394Reader_Triggered;

#else // HAVE_LIDBDC1394
 public:
  CameraImage(void* ptr);
#endif // HAVE_LIBDC1394

 public:
  inline uint8& operator()(indexType x, indexType y, indexType band=0)
    { return pixel(x, y, band); }
  inline const uint8& operator()(indexType x, indexType y, indexType band=0) const
    { return pixel(x, y, band); }
};


#endif // TYPES_CAMERAIMAGE_H

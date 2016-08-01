#ifndef IO_WRITER_H
#define IO_WRITER_H

/**
   \file Writer.h
   Contains definitions of writers that write images/videos to disk.
**/

#include "types/MocaTypes.h"
#include "types/Image8U.h"


/**
   Base class for all writers.
   Each call to putImage() will write an image or a video frame to disk.
**/
class Writer
{
public:
  virtual ~Writer(){}
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void putImage(Image8U const& img) = 0;
};


#endif


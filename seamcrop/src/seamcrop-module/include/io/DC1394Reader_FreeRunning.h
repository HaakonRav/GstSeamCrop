#ifndef IO_DC1394READER_FREERUNNING_H
#define IO_DC1394READER_FREERUNNING_H

#include "io/DC1394Reader.h"

#ifdef HAVE_LIBDC1394


class DC1394Reader_FreeRunning : public DC1394Reader
{
 public:
  DC1394Reader_FreeRunning(Rect dimension, BusSpeed speed, ColorMode color, indexType selectedCamera = 0);
  virtual ~DC1394Reader_FreeRunning();

  void start();
  void stop();

 protected:
  Rect const initialDimension;
};

#endif // HAVE_LIBDC1394

#endif
